# =============================================================================
# Noise-Adaptive ByT5 (TAHIMIK) — The Proposed Model
#
# This is the study's main contribution and the third model variant.
# It extends MrT5's delete gate with a learned noise estimator that
# conditions the compression rate on each input's noise characteristics:
#
#   - Noisier sentences (n > navg) → LESS compression (keep more bytes
#     so the decoder can correct errors)
#   - Cleaner sentences (n < navg) → MORE compression (safely delete
#     redundant bytes for efficiency)
#
# Architecture:
#   Encoder layers [0, gate_layer) →
#       Noise Estimator (mean-pool → MLP → sigmoid → n) →
#       Delete Gate (conditioned on n) →
#   Encoder layers [gate_layer, N) →
#   Decoder
#
# Training signal:
#   L = L_CE + w_rate * L_rate + w_attn_reg * L_attn_reg + L_NE
#
# The noise estimator learns from L_NE only (MSE against n*). Its
# output n is detached before entering the gate so the gate's gradients
# do not leak into the estimator.
# =============================================================================

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, Optional

from src.models.noise_estimator import NoiseEstimator
from src.models.delete_gate import DeleteGate


class NoiseAdaptiveByT5(nn.Module):
    """
    ByT5 with noise-adaptive byte deletion — the TAHIMIK model.

    Integrates the noise estimator and noise-conditioned delete gate
    into the ByT5 encoder. The noise estimator predicts per-sentence
    noise scores that shift the gate's keep/delete threshold.

    Args:
        config: A TAHIMIKConfig instance.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.model_name
        )
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

        # Maximum deletion fraction for the noise-adaptive rate target:
        # d_target(x_i) = d_max * (1 - n_i)
        self.d_max = config.d_max

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
                (batch, seq_len). Added directly to the attention logits as a
                log-space penalty, which is how MrT5 implements soft deletion.

                This must be ADDED to the extended mask, not multiplied into
                the binary mask: HuggingFace converts a binary mask via
                `(1 - mask) * finfo.min`, so a mask value of 0.99997 becomes a
                bias of -1.1e34 and softmax treats it as -inf. Multiplying
                would turn the intended soft mask into a hard one and destroy
                the gradient.
        """
        encoder = self.model.encoder
        # Pass only the two positional arguments. The third parameter is
        # `device` in transformers 4.x but `dtype` in 5.x, so passing a device
        # positionally raises TypeError on 5.x. Two args works on both, and
        # the result already lands on the mask's device.
        extended_mask = encoder.get_extended_attention_mask(
            attention_mask, hidden_states.shape[:2]
        )

        if gate_bias is not None:
            # (batch, seq) -> (batch, 1, 1, seq) to broadcast over heads and
            # query positions, then add as a log-space attention penalty.
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
        noise_level: Optional[torch.Tensor] = None,
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

        Returns:
            Dict with 'loss', 'logits', 'noise_scores', 'gate_outputs',
            'deletion_rate', 'kept_mask', and 'target_deletion_rate'.
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

        # ── Step 3: Noise estimation ────────────────────────────────────
        noise_scores = self.noise_estimator(hidden_states, attention_mask)

        # ── Step 4: Delete gate (noise-conditioned) ─────────────────────
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask, noise_scores=noise_scores
        )

        # Per-sentence adaptive deletion target:
        # d_target(x_i) = d_max * (1 - n_i)
        # Detach noise_scores here so the target doesn't create a gradient
        # path through the noise estimator.
        target_deletion_rate = self.d_max * (1.0 - noise_scores.detach())

        # ── Step 5/6: Apply deletion, then post-gate encoder layers ─────
        if self.training:
            # SOFT deletion. The gate score is added to the attention logits
            # as a log-space penalty, so a byte the gate wants to drop becomes
            # progressively harder to attend to while staying differentiable.
            # The sequence length is unchanged.
            modified_mask = attention_mask
            hidden_states = self._run_encoder_layers(
                hidden_states, attention_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
                gate_bias=gate_outputs.squeeze(-1),
            )
        else:
            # HARD deletion: bytes are physically removed, which is where the
            # computational saving comes from.
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
            "encoder_last_hidden_state": hidden_states,
        }

        if labels is not None:
            decoder_input_ids = self.model._shift_right(labels)
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
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
        else:
            decoder_outputs = self.model.decoder(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            lm_logits = self.model.lm_head(decoder_outputs[0])
            result["logits"] = lm_logits

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

        # Noise estimation + gate
        noise_scores = self.noise_estimator(hidden_states, attention_mask)
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask, noise_scores=noise_scores
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

        # Beam-search decode.
        # encoder_outputs must be a BaseModelOutput, not a bare tuple —
        # generate() reads .last_hidden_state from it to size the beams. Given
        # a tuple it cannot find the encoder states, falls through to the
        # "no input_ids" branch, and raises:
        #   ValueError: `bos_token_id` has to be defined when no `input_ids`...
        return self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states),
            attention_mask=compressed_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
