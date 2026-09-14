# Computes normalized byte-level Levenshtein distance between noisy and clean text.
# n* = edit_distance(noisy_bytes, clean_bytes) / max(len(noisy_bytes), len(clean_bytes))

import editdistance



def compute_noise_level(noisy_text: str, clean_text: str) -> float:
    """
    Compute the noise level n* for a noisy-clean sentence pair.

    The computation operates at the byte level (not character level)
    because ByT5 processes raw UTF-8 bytes. This ensures the noise
    estimator's supervision signal matches the model's input granularity.

    Formula:
        n* = edit_distance(noisy_bytes, clean_bytes) / max(len(noisy_bytes), len(clean_bytes))

    Args:
        noisy_text: The original noisy social media sentence.
        clean_text: The reference normalized sentence.

    Returns:
        Noise level in [0, 1]. Returns 0.0 if both strings are empty.

    Examples:
        >>> compute_noise_level("grabeeee ang init", "grabe ang init")
        # Returns ~0.18 (4 extra bytes out of ~22)
        >>> compute_noise_level("same text", "same text")
        0.0
    """
    noisy_bytes = list(noisy_text.encode("utf-8"))
    clean_bytes = list(clean_text.encode("utf-8"))

    max_len = max(len(noisy_bytes), len(clean_bytes))

    if max_len == 0:
        return 0.0

    distance = editdistance.eval(noisy_bytes, clean_bytes)
    return distance / max_len
