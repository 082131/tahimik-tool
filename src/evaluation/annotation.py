# Inter-annotator agreement using Krippendorff's alpha over edit distance ratios.


import numpy as np
from typing import List, Dict, Optional

try:
    import krippendorff as krippendorff_lib
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False

from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.annotation")


class InterAnnotatorAgreement:
    """
    Computes Krippendorff's alpha for inter-annotator agreement.

    The gold standard dataset has multiple annotators who each normalize
    the same noisy sentence. This class measures how consistently they
    agree on the correct normalization.

    Usage:
        iaa = InterAnnotatorAgreement()

        # Each row = one annotator, each column = one sentence.
        # Values = edit distance ratio between the annotator's normalization
        # and the consensus clean form. np.nan = annotator didn't rate this item.
        annotations = np.array([
            [0.1, 0.2, np.nan, 0.3],
            [0.1, 0.3, 0.4,   0.3],
            [0.2, 0.2, 0.4,   np.nan],
        ])
        result = iaa.compute(annotations)
    """

    def compute(
        self,
        reliability_data: np.ndarray,
        level_of_measurement: str = "ratio",
    ) -> Dict[str, float]:
        """
        Compute Krippendorff's alpha.

        Args:
            reliability_data: 2D array where rows are annotators and
                columns are items. Use np.nan for missing annotations.
                Shape: (num_annotators, num_items)
            level_of_measurement: Scale type. "ratio" for continuous
                edit distance ratios; "nominal" for categorical labels.

        Returns:
            Dict with 'alpha', 'interpretation', 'num_annotators', 'num_items'.
        """
        if not HAS_KRIPPENDORFF:
            logger.warning(
                "krippendorff package not installed. "
                "Install it with: pip install krippendorff"
            )
            return {
                "alpha": None,
                "interpretation": "Package not installed",
                "num_annotators": reliability_data.shape[0],
                "num_items": reliability_data.shape[1],
            }

        alpha = krippendorff_lib.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level_of_measurement,
        )

        # Interpret per Krippendorff's guidelines
        if alpha >= 0.80:
            interpretation = "Reliable agreement (alpha >= 0.80)"
        elif alpha >= 0.67:
            interpretation = "Tentative agreement (0.67 <= alpha < 0.80)"
        else:
            interpretation = "Unreliable agreement (alpha < 0.67)"

        result = {
            "alpha": float(alpha),
            "interpretation": interpretation,
            "num_annotators": int(reliability_data.shape[0]),
            "num_items": int(reliability_data.shape[1]),
        }

        logger.info(
            f"Krippendorff's alpha = {alpha:.4f} — {interpretation}"
        )

        return result

    def compute_from_normalizations(
        self,
        annotator_outputs: List[List[str]],
        reference: List[str],
    ) -> Dict[str, float]:
        """
        Compute agreement from raw annotator normalizations.

        Converts annotator outputs to edit distance ratios against the
        consensus reference, then computes alpha.

        Args:
            annotator_outputs: List of lists, where each inner list is
                one annotator's normalizations. Length of each inner
                list should equal len(reference), or use "" for missing.
            reference: Consensus clean text for each item.

        Returns:
            Same as compute().
        """
        import editdistance

        num_annotators = len(annotator_outputs)
        num_items = len(reference)

        reliability_data = np.full(
            (num_annotators, num_items), np.nan
        )

        for a_idx, outputs in enumerate(annotator_outputs):
            for i_idx, output in enumerate(outputs):
                if not output:
                    continue
                ref = reference[i_idx]
                dist = editdistance.eval(output, ref)
                max_len = max(len(output), len(ref), 1)
                reliability_data[a_idx, i_idx] = dist / max_len

        return self.compute(reliability_data, level_of_measurement="ratio")
