# =============================================================================
# Byte Encoding / Decoding Utilities
#
# ByT5 processes text as raw UTF-8 bytes rather than subword tokens.
# This module handles the conversion between human-readable text and the
# byte-level representation that ByT5 expects.
#
# ByT5 byte vocabulary (from Xue et al., 2022):
#   - Byte IDs 0–2 are reserved: 0 = <pad>, 1 = <eos>, 2 = <unk>
#   - Actual byte values are offset by +3, so byte value b maps to ID b+3
#   - Valid byte IDs range from 3 (byte 0x00) to 258 (byte 0xFF)
# =============================================================================

from typing import List


# ByT5 reserves IDs 0-2 for special tokens; real bytes start at ID 3
_BYT5_OFFSET = 3
_PAD_ID = 0
_EOS_ID = 1


class ByteEncoder:
    """
    Converts text to ByT5-compatible byte IDs and back.

    The ByT5 tokenizer is intentionally bypassed in this implementation
    because byte encoding is deterministic and the HuggingFace tokenizer
    adds unnecessary overhead for byte-level models.
    """

    @staticmethod
    def encode(text: str, max_length: int = 1024) -> List[int]:
        """
        Encode a string into ByT5 byte IDs.

        Each UTF-8 byte is offset by +3 to avoid collision with the
        special token IDs (pad=0, eos=1, unk=2). The sequence is
        truncated to max_length and terminated with <eos>.

        Args:
            text: Input string to encode.
            max_length: Maximum number of byte IDs (including <eos>).

        Returns:
            List of integer byte IDs, ending with <eos>.
        """
        raw_bytes = text.encode("utf-8")

        # Reserve one position for <eos>
        truncated = raw_bytes[: max_length - 1]

        byte_ids = [b + _BYT5_OFFSET for b in truncated]
        byte_ids.append(_EOS_ID)

        return byte_ids

    @staticmethod
    def decode(byte_ids: List[int]) -> str:
        """
        Decode ByT5 byte IDs back into a string.

        Special tokens (pad, eos, unk) are stripped. Invalid byte
        sequences are replaced with the Unicode replacement character.

        Args:
            byte_ids: List of ByT5 byte IDs.

        Returns:
            Decoded UTF-8 string.
        """
        raw_bytes = []

        for bid in byte_ids:
            # Skip special tokens
            if bid <= 2:
                continue
            raw_bytes.append(bid - _BYT5_OFFSET)

        return bytes(raw_bytes).decode("utf-8", errors="replace")

    @staticmethod
    def pad_sequence(
        byte_ids: List[int], max_length: int
    ) -> tuple[List[int], List[int]]:
        """
        Pad a byte ID sequence and produce an attention mask.

        Args:
            byte_ids: Encoded byte IDs (already includes <eos>).
            max_length: Target padded length.

        Returns:
            Tuple of (padded_ids, attention_mask) where attention_mask
            is 1 for real tokens and 0 for padding.
        """
        seq_len = len(byte_ids)

        if seq_len >= max_length:
            padded = byte_ids[:max_length]
            mask = [1] * max_length
        else:
            pad_len = max_length - seq_len
            padded = byte_ids + [_PAD_ID] * pad_len
            mask = [1] * seq_len + [0] * pad_len

        return padded, mask

    @staticmethod
    def byte_length(text: str) -> int:
        """Return the UTF-8 byte length of a string."""
        return len(text.encode("utf-8"))
