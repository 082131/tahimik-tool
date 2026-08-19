# =============================================================================
# MrT5 Fixed-Rate Compression Baseline
#
# The second model variant. It inserts a learned delete gate after encoder
# layer 3 that compresses the byte sequence at a FIXED target rate (50%)
# regardless of input noise. This demonstrates the efficiency gains of
# byte-level compression but also its limitation: clean and noisy inputs
# receive the same compression, which can hurt accuracy on noisy text.
#
# Architecture:
#   Encoder layers 0-2 → Delete Gate → Encoder layers 3-N → Decoder
#
# The gate uses soft deletion during training (gate outputs as attention
# masks) and hard deletion during inference (physical byte removal).
#
# Reference: Kallini et al. (2025), "MrT5: Dynamic Token Merging for
#            Efficient Byte-Level Language Models" (ICLR 2025).
# =============================================================================

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Dict, Optional

from src.models.delete_gate import DeleteGate


class FixedCompressionByT5(nn.Module):
    """
    ByT5 with fixed-rate byte deletion (MrT5 replication).

    The delete gate sits after encoder layer `delete_gate_layer` and
    removes a fixed fraction of bytes. The target deletion rate is
    enforced by a rate loss term during training.

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
        )

        # Fixed deletion target for the rate loss
        self.fixed_deletion_target = config.fixed_deletion_target

    def _run_encoder_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        start_layer: int,
        end_layer: int,
        gate_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run a subset of encoder layers on the hidden states.

        Args:
            gate_bias: Optional per-byte gate scores in [k, 0], shape
                (batch, seq_len). Added to the attention logits as a log-space
                penalty — the way MrT5 implements soft deletion.

                It must be ADDED to the extended mask, not multiplied into the
                binary mask: HuggingFace builds the additive mask as
                `(1 - mask) * finfo.min`, so a mask of 0.99997 becomes a bias
                of -1.1e34, which softmax reads as -inf. Multiplying would
                make the "soft" mask hard and kill the gradient.
        """
        encoder = self.model.encoder
        extended_mask = encoder.get_extended_attention_mask(
            attention_mask, hidden_states.shape[:2], hidden_states.device
        )

        if gate_bias is not None:
            extended_mask = extended_mask + gate_bias[:, None, None, :]

        for i in range(start_layer, end_layer):
            layer = encoder.block[i]
            layer_output = layer(
                hidden_states,
                attention_mask=extended_mask,
            )
            hidden_states = layer_output[0]

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
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
            and 'kept_mask'.
        """
        encoder = self.model.encoder
        num_layers = len(encoder.block)

        # ── Step 1: Embedding ───────────────────────────────────────────
        inputs_embeds = encoder.embed_tokens(input_ids)
        hidden_states = encoder.dropout(inputs_embeds)

        # ── Step 2: Pre-gate encoder layers ─────────────────────────────
        hidden_states = self._run_encoder_layers(
            hidden_states, attention_mask,
            start_layer=0,
            end_layer=self.delete_gate_layer,
        )

        # ── Step 3: Delete gate ─────────────────────────────────────────
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask
        )

        # ── Step 4/5: Apply deletion, then post-gate encoder layers ─────
        if self.training:
            # SOFT deletion: gate score added to attention logits as a
            # log-space penalty. Differentiable; length unchanged.
            modified_mask = attention_mask
            hidden_states = self._run_encoder_layers(
                hidden_states, attention_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
                gate_bias=gate_outputs.squeeze(-1),
            )
        else:
            # HARD deletion: bytes physically removed — the actual speedup.
            hidden_states, modified_mask = self.delete_gate.apply_hard_deletion(
                hidden_states, kept_mask
            )
            hidden_states = self._run_encoder_layers(
                hidden_states, modified_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
            )

        # Final layer norm
        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)

        if self.training:
            # Suppress deleted bytes in what the decoder sees. Applied after
            # the final norm, since RMS normalisation would undo it.
            hidden_states = hidden_states * keep_prob.unsqueeze(-1)

        # ── Step 6: Decoder ─────────────────────────────────────────────
        encoder_outputs = (hidden_states,)

        if labels is not None:
            decoder_input_ids = self.model._shift_right(labels)
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            sequence_output = decoder_outputs[0]
            lm_logits = self.model.lm_head(sequence_output)

            result = {
                "logits": lm_logits,
                "gate_outputs": gate_outputs,
                "keep_prob": keep_prob,
                "kept_mask": kept_mask,
                "deletion_rate": deletion_rate,
                "encoder_last_hidden_state": hidden_states,
            }

            # Cross-entropy loss (computed by the external loss module)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result["loss"] = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
            )

            return result
        else:
            decoder_outputs = self.model.decoder(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            lm_logits = self.model.lm_head(decoder_outputs[0])

            return {
                "logits": lm_logits,
                "gate_outputs": gate_outputs,
                "keep_prob": keep_prob,
                "kept_mask": kept_mask,
                "deletion_rate": deletion_rate,
                "encoder_last_hidden_state": hidden_states,
            }

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
        self.eval()

        encoder = self.model.encoder
        num_layers = len(encoder.block)

        # Embedding + pre-gate layers
        inputs_embeds = encoder.embed_tokens(input_ids)
        hidden_states = encoder.dropout(inputs_embeds)
        hidden_states = self._run_encoder_layers(
            hidden_states, attention_mask,
            start_layer=0,
            end_layer=self.delete_gate_layer,
        )

        # Delete gate (hard mode in eval)
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask
        )
        hidden_states, compressed_mask = self.delete_gate.apply_hard_deletion(
            hidden_states, kept_mask
        )

        # Post-gate layers + final norm
        hidden_states = self._run_encoder_layers(
            hidden_states, compressed_mask,
            start_layer=self.delete_gate_layer,
            end_layer=num_layers,
        )
        hidden_states = encoder.final_layer_norm(hidden_states)

        # Beam-search decode
        encoder_outputs = (hidden_states,)

        return self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=compressed_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
