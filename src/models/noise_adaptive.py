# Noise-Adaptive ByT5 (TAHIMIK)
# Dynamically adjusts delete gate thresholds based on predicted sentence noise.
# Encoder layers [0, gate_layer) -> Noise Estimator + Delete Gate -> Encoder layers [gate_layer, N) -> Decoder.

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, Optional

from src.models.noise_estimator import NoiseEstimator
from src.models.delete_gate import (
    DeleteGate,
    disable_embedded_mrt5_gate,
    get_embedded_mrt5_gate,
    load_mrt5_pretrained_gate,
)
from src.models.encoder_layers import (
    run_encoder_layers,
    compress_position_bias,
)


class NoiseAdaptiveByT5(nn.Module):
    """
    ByT5 model integrating a noise estimator and noise-conditioned delete gate.

    Args:
        config: A TAHIMIKConfig instance.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        # TAHIMIK begins from Stanford MrT5 Small; its own contribution is
        # the adaptive gate modification below, not a replacement backbone.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        source_gate = get_embedded_mrt5_gate(self.model)
        disable_embedded_mrt5_gate(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        hidden_dim = self.model.config.d_model
        self.delete_gate_layer = config.delete_gate_layer

        # ── Noise Estimator ─────────────────────────────────────────────
        self.noise_estimator = NoiseEstimator(
            hidden_dim=hidden_dim,
            intermediate_dim=config.noise_estimator_hidden_dim,
        )

        # ── Delete Gate (noise-adaptive mode) ───────────────────────────
        self.delete_gate = DeleteGate(
            hidden_dim=hidden_dim,
            k=config.gate_k,
            noise_adaptive=True,
            noise_avg_momentum=config.noise_avg_momentum,
            use_gumbel_noise=getattr(config, "use_gumbel_noise", True),
        )
        if not load_mrt5_pretrained_gate(
            self.delete_gate,
            config.model_name,
            source_gate=source_gate,
        ):
            raise RuntimeError(
                "Could not load the Stanford MrT5 delete gate; refusing to use a random gate."
            )

        # Maximum deletion fraction for the noise-adaptive rate target:
        # d_target(x_i) = d_max * (1 - n_i)
        self.d_max = config.d_max

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with noise-adaptive byte deletion.

        Flow:
          1. Embed input bytes.
          2. Run pre-gate encoder layers [0, gate_layer).
          3. Noise estimator predicts n from the hidden states.
          4. Delete gate uses n to condition keep/delete scores.
          5. Apply soft (train) or hard (eval) deletion.
          6. Run post-gate encoder layers [gate_layer, N).
          7. Decoder generates normalized output.

        Args:
            input_ids: Byte-level input IDs. Shape: (batch, seq_len)
            attention_mask: Binary mask. Shape: (batch, seq_len)
            labels: Target byte IDs with -100 padding. Shape: (batch, target_len)
            noise_level: Ground-truth n* for L_NE supervision. Shape: (batch,)
            decoder_input_ids: Optional explicit decoder input IDs.

        Returns:
            Dict with 'loss', 'logits', 'noise_scores', 'gate_outputs',
            'deletion_rate', 'kept_mask', 'target_deletion_rate',
            'adaptive_coefficient', 'noise_average', and 'gate_attention_mask'.
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

        # ── Step 3: Noise estimation ────────────────────────────────────
        noise_scores = self.noise_estimator(hidden_states, attention_mask)

        # Compute dynamic target deletion rate: d_target = d_max * (1 - n)
        # Detach noise scores so L_rate doesn't backprop into the noise estimator
        target_deletion_rate = self.d_max * (1.0 - noise_scores.detach())

        # ── Step 4: Delete gate with noise-adaptive conditioning ────────
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask, noise_scores=noise_scores
        )

        # ── Step 5/6: Apply deletion, then post-gate encoder layers ─────
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
            # HARD deletion: bytes are physically removed, which is where the
            # computational saving comes from.
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
            # Suppress deleted bytes in what the decoder cross-attends to.
            # Applied AFTER the final norm — T5 uses RMS normalisation, which
            # would otherwise rescale the suppression straight back out.
            hidden_states = hidden_states * keep_prob.unsqueeze(-1)

        # ── Step 7: Decoder ─────────────────────────────────────────────
        result = {
            "noise_scores": noise_scores,
            "gate_outputs": gate_outputs,
            "keep_prob": keep_prob,
            "kept_mask": kept_mask,
            "deletion_rate": deletion_rate,
            "target_deletion_rate": target_deletion_rate,
            "adaptive_coefficient": self.delete_gate.adaptive_coefficient,
            "noise_average": self.delete_gate.noise_avg,
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

            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result["loss"] = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
            )

            # Noise estimator loss: MSE(predicted_n, ground_truth_n*)
            if noise_level is not None:
                ne_loss = nn.functional.mse_loss(noise_scores, noise_level)
                result["ne_loss"] = ne_loss
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
        Generate normalized text with noise-adaptive hard deletion.

        At inference time:
          1. Pre-gate layers produce contextual hidden states.
          2. The noise estimator predicts the sentence's noise level.
          3. The gate uses the noise level to decide which bytes to delete.
          4. Bytes are physically removed (hard deletion).
          5. Post-gate layers process the shorter sequence.
          6. The decoder generates via beam search.
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

        # Noise estimation + gate
        noise_scores = self.noise_estimator(hidden_states, attention_mask)
        target_deletion_rate = self.d_max * (1.0 - noise_scores.detach())
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask, noise_scores=noise_scores
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
        noise_values = noise_scores.detach().cpu().tolist()
        target_values = target_deletion_rate.detach().cpu().tolist()
        telemetry = []
        for input_count, kept_count, noise_score, target_rate in zip(
            input_counts, kept_counts, noise_values, target_values
        ):
            input_count = int(input_count)
            kept_count = int(kept_count)
            deleted_count = input_count - kept_count
            telemetry.append(
                {
                    "compression_mode": "adaptive",
                    "input_token_count": input_count,
                    "kept_token_count": kept_count,
                    "deleted_token_count": deleted_count,
                    "deletion_rate": deleted_count / input_count if input_count else 0.0,
                    "noise_score": float(noise_score),
                    "target_deletion_rate": float(target_rate),
                }
            )

        return (
            BaseModelOutput(last_hidden_state=hidden_states),
            compressed_mask,
            telemetry,
        )
