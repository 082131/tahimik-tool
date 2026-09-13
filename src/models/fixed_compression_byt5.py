# ByT5 with fixed-rate byte deletion gate (MrT5).
# Compresses byte sequences at a fixed target rate (default 50%).
# Uses soft masking during training and hard sequence pruning during inference.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, Optional

from src.models.delete_gate import DeleteGate
from src.models.encoder_layers import (
    run_encoder_layers,
    compress_position_bias,
)


class FixedCompressionByT5(nn.Module):
    """
    ByT5 model with mid-encoder fixed-rate byte deletion.

    Args:
        config: An MrT5Config instance.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Gate placement: after this encoder layer
        self.delete_gate_layer = config.delete_gate_layer

        # The delete gate in non-adaptive mode (fixed rate)
        hidden_dim = self.model.config.d_model
        self.delete_gate = DeleteGate(
            hidden_dim=hidden_dim,
            k=config.gate_k,
            noise_adaptive=False,
            use_gumbel_noise=getattr(config, "use_gumbel_noise", True),
        )

        # Fixed deletion target for the rate loss
        self.fixed_deletion_target = config.fixed_deletion_target

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with fixed-rate byte deletion.

        The encoder is split at delete_gate_layer:
          1. Layers [0, gate_layer) produce contextual representations.
          2. The delete gate scores and masks/removes bytes.
          3. Layers [gate_layer, N) process the compressed sequence.
          4. The decoder generates the normalized output.

        Returns:
            Dict with 'loss', 'logits', 'gate_outputs', 'deletion_rate',
            'kept_mask', and 'gate_attention_mask'.
        """
        encoder = self.model.encoder
        num_layers = len(encoder.block)

        # ── Step 1: Embedding ───────────────────────────────────────────
        inputs_embeds = encoder.embed_tokens(input_ids)
        hidden_states = encoder.dropout(inputs_embeds)

        # ── Step 2: Pre-gate encoder layers (threads position_bias) ─────
        pre_gate_res = run_encoder_layers(
            encoder,
            hidden_states,
            attention_mask,
            start_layer=0,
            end_layer=self.delete_gate_layer,
        )
        hidden_states = pre_gate_res.hidden_states
        position_bias = pre_gate_res.position_bias

        # ── Step 3: Delete gate ─────────────────────────────────────────
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask
        )

        # ── Step 4/5: Apply deletion, then post-gate encoder layers ─────
        if self.training:
            # SOFT deletion: gate score added to attention logits as a
            # log-space penalty. Differentiable; length unchanged.
            modified_mask = attention_mask
            post_gate_res = run_encoder_layers(
                encoder,
                hidden_states,
                attention_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
                position_bias=position_bias,
                gate_bias=gate_outputs.squeeze(-1),
            )
            hidden_states = post_gate_res.hidden_states
        else:
            # HARD deletion: bytes physically removed — the actual speedup.
            del_result = self.delete_gate.apply_hard_deletion(
                hidden_states, kept_mask
            )
            hidden_states = del_result.hidden_states
            modified_mask = del_result.attention_mask

            # Compress pre-gate position bias to match the pruned sequence length
            if position_bias is not None:
                post_gate_bias = compress_position_bias(
                    position_bias, del_result.source_positions, modified_mask
                )
            else:
                post_gate_bias = None

            post_gate_res = run_encoder_layers(
                encoder,
                hidden_states,
                modified_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
                position_bias=post_gate_bias,
            )
            hidden_states = post_gate_res.hidden_states

        # Final layer norm
        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)

        if self.training:
            # Suppress deleted bytes in what the decoder sees. Applied after
            # the final norm, since RMS normalisation would undo it.
            hidden_states = hidden_states * keep_prob.unsqueeze(-1)

        # ── Step 6: Decoder ─────────────────────────────────────────────
        result = {
            "gate_outputs": gate_outputs,
            "keep_prob": keep_prob,
            # Emitted so the loss uses THIS variant's configured
            # target rather than silently falling back to 0.5.
            "fixed_deletion_target": self.fixed_deletion_target,
            "kept_mask": kept_mask,
            "deletion_rate": deletion_rate,
            "gate_attention_mask": attention_mask,
            "encoder_last_hidden_state": hidden_states,
        }

        if labels is not None:
            dec_input_ids = self.model._shift_right(labels)
            decoder_outputs = self.model.decoder(
                input_ids=dec_input_ids,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            sequence_output = decoder_outputs[0]
            lm_logits = self.model.lm_head(sequence_output)
            result["logits"] = lm_logits

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result["loss"] = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
            )
        elif decoder_input_ids is not None:
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            sequence_output = decoder_outputs[0]
            lm_logits = self.model.lm_head(sequence_output)
            result["logits"] = lm_logits
        else:
            raise ValueError(
                "Either 'labels' or 'decoder_input_ids' must be provided to forward(). "
                "For free text generation, call model.generate() instead."
            )

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 1024,
        num_beams: int = 4,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate normalized text with hard byte deletion.

        The encoder runs with hard deletion (physically shorter sequences),
        then the decoder generates via beam search over the compressed
        encoder output.
        """
        encoder_outputs, compressed_mask, _ = self._prepare_hard_deletion_encoder(
            input_ids, attention_mask
        )
        return self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=compressed_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    def generate_with_telemetry(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 1024,
        num_beams: int = 4,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Generate text and report the hard-deletion mask actually used."""
        encoder_outputs, compressed_mask, telemetry = self._prepare_hard_deletion_encoder(
            input_ids, attention_mask
        )
        generated = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=compressed_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
        return generated, telemetry

    def _prepare_hard_deletion_encoder(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[BaseModelOutput, torch.Tensor, list[dict]]:
        """Return the compressed encoder output and factual per-item counts."""
        self.eval()

        encoder = self.model.encoder
        num_layers = len(encoder.block)

        # Embedding + pre-gate layers
        inputs_embeds = encoder.embed_tokens(input_ids)
        hidden_states = encoder.dropout(inputs_embeds)
        pre_gate_res = run_encoder_layers(
            encoder,
            hidden_states,
            attention_mask,
            start_layer=0,
            end_layer=self.delete_gate_layer,
        )
        hidden_states = pre_gate_res.hidden_states
        position_bias = pre_gate_res.position_bias

        # Delete gate (hard mode in eval)
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask
        )
        del_result = self.delete_gate.apply_hard_deletion(
            hidden_states, kept_mask
        )
        hidden_states = del_result.hidden_states
        compressed_mask = del_result.attention_mask

        # Compress pre-gate position bias to match the pruned sequence length
        if position_bias is not None:
            post_gate_bias = compress_position_bias(
                position_bias, del_result.source_positions, compressed_mask
            )
        else:
            post_gate_bias = None

        # Post-gate layers + final norm
        post_gate_res = run_encoder_layers(
            encoder,
            hidden_states,
            compressed_mask,
            start_layer=self.delete_gate_layer,
            end_layer=num_layers,
            position_bias=post_gate_bias,
        )
        hidden_states = encoder.final_layer_norm(post_gate_res.hidden_states)

        input_counts = attention_mask.sum(dim=1).detach().cpu().tolist()
        kept_counts = compressed_mask.sum(dim=1).detach().cpu().tolist()
        telemetry = []
        for input_count, kept_count in zip(input_counts, kept_counts):
            input_count = int(input_count)
            kept_count = int(kept_count)
            deleted_count = input_count - kept_count
            telemetry.append(
                {
                    "compression_mode": "fixed",
                    "input_token_count": input_count,
                    "kept_token_count": kept_count,
                    "deleted_token_count": deleted_count,
                    "deletion_rate": deleted_count / input_count if input_count else 0.0,
                }
            )

        return (
            BaseModelOutput(last_hidden_state=hidden_states),
            compressed_mask,
            telemetry,
        )
