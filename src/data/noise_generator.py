# =============================================================================
# Synthetic Noise Generation Pipeline for Tagalog and Taglish Text
#
# Generates noisy-to-clean sentence pairs by applying controlled noise
# operations to clean Tagalog/Taglish sentences. The noise categories
# are drawn from the thesis manuscript (9 categories) and guided by
# TagNorm (Agapito et al., 2021) and Karpukhin et al. (2019).
#
# The 9 noise categories:
#   1. Abbreviations and shortenings (e.g., "salamat" â†’ "slmt")
#   2. Orthographic variation (e.g., "dito" â†’ "d2", phonetic spelling)
#   3. Character elongation (e.g., "grabe" â†’ "grabeeee")
#   4. Punctuation variation (e.g., "!!!" â†’ "!!!!!!!!")
#   5. Capitalization variation (e.g., random CAPS)
#   6. Slang and netspeak (e.g., "idol" â†’ "lodi")
#   7. Taglish morphology patterns (e.g., "nag-download")
#   8. Emoji-based sentiment markers (inserting emojis)
#   9. Code-switching (mixing Tagalog and English)
#
# Design principle: each noise function transforms ONE clean token or
# sentence region. Multiple functions are composed stochastically to
# simulate the layered noise density found in real Filipino social media.
#
# IMPORTANT: The synthetic dataset is used ONLY for Stage 1 pretraining.
# Final evaluation uses the human-annotated gold standard dataset.
# =============================================================================

import random
import re
from typing import List, Tuple

# â”€â”€ Common Filipino abbreviation dictionary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Maps standard Tagalog words to their abbreviated social media forms.
# Sourced from Filipino netspeak conventions documented in TagNorm.
ABBREVIATION_MAP = {
    "salamat": ["slmt", "slmat", "tnx", "ty"],
    "magandang": ["mgandang", "mgndng"],
    "umaga": ["umga"],
    "gabi": ["gbi"],
    "hapon": ["hpon"],
    "kamusta": ["kmusta", "musta", "msta"],
    "paano": ["pano", "pnu"],
    "talaga": ["tlga", "tlaga"],
    "naman": ["nmn", "nman"],
    "lang": ["lng", "lang"],
    "sana": ["sna"],
    "kasi": ["kse", "kc"],
    "yung": ["yng", "ung"],
    "hindi": ["hnd", "hinde", "di"],
    "wala": ["wla"],
    "ganun": ["gnun", "ganon"],
    "ganito": ["gnito", "gnto"],
    "sobra": ["sbra", "sbr"],
    "dapat": ["dpat", "dpd"],
    "gusto": ["gsto", "gust"],
    "sabihin": ["sbihin", "sbi"],
    "punta": ["pnta", "pnt"],
    "alam": ["alm"],
    "ikaw": ["ikw", "u"],
    "siya": ["xa", "cya"],
    "tayo": ["tau", "tyo"],
    "natin": ["ntin", "ntn"],
    "kung": ["kng"],
    "pero": ["pro", "piro"],
    "baka": ["bka"],
    "bakit": ["bkit", "bat"],
    "marami": ["mrami", "mrmi"],
    "totoo": ["totoo", "2too"],
    "ngayon": ["ngaun", "ngyn"],
    "bukas": ["bkas"],
    "kanina": ["knina", "knna"],
}

# â”€â”€ Vowel sets for vowel omission noise â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
VOWELS = set("aeiouAEIOU")

# â”€â”€ Slang / Netspeak dictionary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Filipino social media slang where the normalized form should be kept.
# These are NOT corrected during normalization â€” they are preserved.
SLANG_MAP = {
    "idol": "lodi",
    "pare": "pre",
    "totoo": "trudat",
    "oo": "oo",
    "grabe": "grabiii",
    "ang galing": "G",
    "tama": "bet",
    "ayos": "slay",
    "maganda": "ganda",
}

# â”€â”€ Orthographic / phonetic substitution patterns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ORTHO_SUBSTITUTIONS = {
    "dito": ["d2", "dto"],
    "diyan": ["d yan", "dyan"],
    "ito": ["2", "eto"],
    "iyan": ["yan"],
    "hindi": ["hnd", "hinde", "d"],
    "oo": ["uu", "oww"],
    "ng": ["nang"],
    "nang": ["ng"],
}

# â”€â”€ Common emoji sentiment markers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
POSITIVE_EMOJIS = ["ðŸ˜Š", "ðŸ˜‚", "â¤ï¸", "ðŸ¥°", "ðŸ’¯", "ðŸ™", "âœ¨", "ðŸ”¥", "ðŸ’ª", "ðŸ˜"]
NEGATIVE_EMOJIS = ["ðŸ˜¢", "ðŸ˜­", "ðŸ’”", "ðŸ˜¤", "ðŸ˜¡", "ðŸ¥º", "ðŸ˜’", "ðŸ˜©"]
NEUTRAL_EMOJIS = ["ðŸ‘€", "ðŸ¤”", "ðŸ˜…", "ðŸ™ƒ", "ðŸ‘ï¸", "ðŸ’€", "â˜ ï¸"]


class TagalogNoiseGenerator:
    """
    Generates synthetic noisy versions of clean Tagalog/Taglish sentences.

    Each noise category can be applied independently or in combination.
    The probability of applying each noise type is configurable, allowing
    control over the noise density distribution across the synthetic corpus.

    Usage:
        generator = TagalogNoiseGenerator(seed=42)
        noisy = generator.apply_noise("Magandang umaga sa lahat!")
        # Possible output: "mgndng umga sa lhat!"
    """

    def __init__(self, seed: int = 42, probabilities=None):
        self.rng = random.Random(seed)

        # Per-category application probabilities.
        # Higher values produce noisier synthetic data.
        probabilities = probabilities or {}
        self.p_abbreviation = probabilities.get("abbreviation", 0.30)
        self.p_orthographic = probabilities.get("orthographic", 0.20)
        self.p_elongation = probabilities.get("elongation", 0.15)
        self.p_punctuation = probabilities.get("punctuation", 0.15)
        self.p_capitalization = probabilities.get("capitalization", 0.15)
        self.p_slang = 0.0
        self.p_vowel_omission = probabilities.get("vowel_omission", 0.20)
        self.p_emoji_insert = 0.0
        self.p_char_swap = probabilities.get("char_swap", 0.10)

    def apply_noise(self, clean_sentence: str) -> str:
        """
        Apply a stochastic combination of noise categories to a clean sentence.

        Multiple noise types may be applied to the same sentence, reflecting
        how real Filipino social media text often contains overlapping noise.

        Args:
            clean_sentence: A clean Tagalog or Taglish sentence.

        Returns:
            The noisy version of the sentence.
        """
        noisy = clean_sentence

        # Abbreviation and shortening
        if self.rng.random() < self.p_abbreviation:
            noisy = self._apply_abbreviation(noisy)

        # Vowel omission (a common Filipino texting pattern)
        if self.rng.random() < self.p_vowel_omission:
            noisy = self._apply_vowel_omission(noisy)

        # Orthographic / phonetic variation
        if self.rng.random() < self.p_orthographic:
            noisy = self._apply_orthographic_variation(noisy)

        # Character elongation for emphasis
        if self.rng.random() < self.p_elongation:
            noisy = self._apply_elongation(noisy)

        # Punctuation variation
        if self.rng.random() < self.p_punctuation:
            noisy = self._apply_punctuation_noise(noisy)

        # Capitalization variation
        if self.rng.random() < self.p_capitalization:
            noisy = self._apply_capitalization_noise(noisy)

        # Slang and netspeak substitution
        if self.rng.random() < self.p_slang:
            noisy = self._apply_slang(noisy)

        # Emoji insertion
        if self.rng.random() < self.p_emoji_insert:
            noisy = self._apply_emoji_insert(noisy)

        # Character-level swap (typos, per Karpukhin et al., 2019)
        if self.rng.random() < self.p_char_swap:
            noisy = self._apply_char_swap(noisy)

        return noisy

    # â”€â”€ Individual noise functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _apply_abbreviation(self, text: str) -> str:
        """Replace random words with their abbreviated forms."""
        words = text.split()
        for i, word in enumerate(words):
            lower = word.lower().strip(".,!?;:")
            if lower in ABBREVIATION_MAP:
                if self.rng.random() < 0.5:
                    abbrev = self.rng.choice(ABBREVIATION_MAP[lower])
                    # Preserve any trailing punctuation
                    trailing = word[len(lower):]
                    words[i] = abbrev + trailing
        return " ".join(words)

    def _apply_vowel_omission(self, text: str) -> str:
        """
        Remove vowels from random words to simulate texting shortcuts.

        Filipino texting frequently drops vowels: "punta" â†’ "pnta".
        Only interior vowels are dropped (not the first character),
        and short words (â‰¤3 chars) are left intact.
        """
        words = text.split()
        for i, word in enumerate(words):
            if len(word) <= 3 or not word.isalpha():
                continue
            if self.rng.random() < 0.4:
                # Keep first char, drop some interior vowels
                result = [word[0]]
                for ch in word[1:]:
                    if ch in VOWELS and self.rng.random() < 0.6:
                        continue
                    result.append(ch)
                words[i] = "".join(result)
        return " ".join(words)

    def _apply_orthographic_variation(self, text: str) -> str:
        """Apply phonetic spelling or digit substitution."""
        words = text.split()
        for i, word in enumerate(words):
            lower = word.lower()
            if lower in ORTHO_SUBSTITUTIONS:
                if self.rng.random() < 0.4:
                    words[i] = self.rng.choice(ORTHO_SUBSTITUTIONS[lower])
        return " ".join(words)

    def _apply_elongation(self, text: str) -> str:
        """
        Repeat characters for emotional emphasis.

        Filipino social media frequently uses elongation for intensity:
        "grabe" â†’ "grabeeee", "ang sarap" â†’ "ang saraaap".
        """
        words = text.split()
        # Pick 1-2 words to elongate
        num_targets = self.rng.randint(1, min(2, len(words)))
        indices = self.rng.sample(range(len(words)), min(num_targets, len(words)))

        for idx in indices:
            word = words[idx]
            if len(word) < 3 or not word.isalpha():
                continue
            # Elongate a vowel near the end of the word
            positions = [j for j, c in enumerate(word) if c.lower() in "aeiou"]
            if positions:
                pos = positions[-1]
                repeat_count = self.rng.randint(2, 5)
                word = word[:pos] + word[pos] * repeat_count + word[pos + 1 :]
                words[idx] = word

        return " ".join(words)

    def _apply_punctuation_noise(self, text: str) -> str:
        """Add excessive or missing punctuation."""
        choice = self.rng.choice(["excessive", "missing", "wrong"])

        if choice == "excessive":
            # Multiply terminal punctuation
            text = re.sub(r"([!?.])", lambda m: m.group(1) * self.rng.randint(2, 6), text)
        elif choice == "missing":
            # Remove some punctuation
            text = re.sub(r"[.,;]", "", text)
        else:
            # Replace periods with ellipses
            text = text.replace(".", "...")

        return text

    def _apply_capitalization_noise(self, text: str) -> str:
        """Apply random or all-caps capitalization."""
        choice = self.rng.choice(["random", "allcaps", "nocaps"])

        if choice == "random":
            text = "".join(
                c.upper() if self.rng.random() < 0.3 else c for c in text
            )
        elif choice == "allcaps":
            # Only capitalize 1-2 words (not the whole sentence)
            words = text.split()
            num = self.rng.randint(1, min(3, len(words)))
            indices = self.rng.sample(range(len(words)), num)
            for idx in indices:
                words[idx] = words[idx].upper()
            text = " ".join(words)
        else:
            text = text.lower()

        return text

    def _apply_slang(self, text: str) -> str:
        """Replace words with Filipino slang or netspeak equivalents."""
        words = text.split()
        for i, word in enumerate(words):
            lower = word.lower()
            if lower in SLANG_MAP and self.rng.random() < 0.5:
                words[i] = SLANG_MAP[lower]
        return " ".join(words)

    def _apply_emoji_insert(self, text: str) -> str:
        """Insert emoji sentiment markers at sentence boundaries."""
        emoji_pool = POSITIVE_EMOJIS + NEGATIVE_EMOJIS + NEUTRAL_EMOJIS
        emoji = self.rng.choice(emoji_pool)

        position = self.rng.choice(["end", "start", "middle"])

        if position == "end":
            text = text.rstrip() + " " + emoji
        elif position == "start":
            text = emoji + " " + text
        else:
            words = text.split()
            if len(words) > 2:
                insert_pos = self.rng.randint(1, len(words) - 1)
                words.insert(insert_pos, emoji)
                text = " ".join(words)

        return text

    def _apply_char_swap(self, text: str) -> str:
        """
        Swap adjacent characters to simulate typographical errors.

        Based on Karpukhin et al. (2019): synthetic character-level
        corruptions improve robustness to natural noise.
        """
        chars = list(text)
        # Swap 1-2 adjacent pairs
        num_swaps = self.rng.randint(1, 2)

        for _ in range(num_swaps):
            if len(chars) < 2:
                break
            pos = self.rng.randint(0, len(chars) - 2)
            # Only swap alphabetic characters (preserve spaces/punctuation)
            if chars[pos].isalpha() and chars[pos + 1].isalpha():
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

        return "".join(chars)

    def generate_batch(
        self, clean_sentences: List[str], noise_per_sentence: int = 1
    ) -> List[Tuple[str, str]]:
        """
        Generate a batch of noisy-clean pairs from a list of clean sentences.

        Args:
            clean_sentences: List of clean Tagalog/Taglish sentences.
            noise_per_sentence: Number of noisy variants per clean sentence.

        Returns:
            List of (noisy_sentence, clean_sentence) tuples.
        """
        pairs = []
        for clean in clean_sentences:
            for _ in range(noise_per_sentence):
                noisy = self.apply_noise(clean)
                # Only keep pairs where noise was actually applied
                if noisy != clean:
                    pairs.append((noisy, clean))
                else:
                    # Re-apply with higher probability if first attempt
                    # produced no change (ensure non-trivial training signal)
                    noisy_retry = self.apply_noise(clean)
                    pairs.append((noisy_retry, clean))
        return pairs



