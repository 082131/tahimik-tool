import pytest
import torch
from pathlib import Path

from src.data.dataset import NormalizationDataset
from src.data.noise_label import compute_noise_level
from src.data.noise_policy import SyntheticPairLineage, build_probability_manifest
from src.data.preprocessing import DataPipeline, prepare_paired_examples


class ByteTokenizer:
    """Test tokenizer: UTF-8 bytes plus one EOS position, never truncating."""

    pad_token_id = 0

    def __call__(
        self,
        text,
        max_length=None,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=True,
    ):
        if truncation:
            raise AssertionError("paired preparation must not request tokenizer truncation")
        token_ids = list(text.encode("utf-8"))
        if add_special_tokens:
            token_ids.append(1)
        attention_mask = [1] * len(token_ids)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": token_ids, "attention_mask": attention_mask}


def test_policy_drops_overflowing_aligned_word_group_and_relabels_pair():
    prepared = prepare_paired_examples(
        ["aang gandaaaa"],
        ["ang ganda"],
        tokenizer=ByteTokenizer(),
        max_input_length=5,
        max_target_length=4,
    )

    assert prepared.noisy_texts == ["aang"]
    assert prepared.clean_texts == ["ang"]
    assert prepared.noise_levels == [pytest.approx(compute_noise_level("aang", "ang"))]
    assert prepared.source_indices == [0]
    assert prepared.audit.transformed_pairs == 1
    assert prepared.audit.unchanged_pairs == 0
    assert prepared.audit.excluded_pairs == 0


def test_policy_keeps_split_merge_as_one_aligned_group():
    prepared = prepare_paired_examples(
        ["sanaol ngayon"],
        ["sana all ngayon"],
        tokenizer=ByteTokenizer(),
        max_input_length=7,
        max_target_length=9,
    )

    assert prepared.noisy_texts == ["sanaol"]
    assert prepared.clean_texts == ["sana all"]


def test_policy_excludes_pair_when_first_aligned_group_cannot_fit():
    prepared = prepare_paired_examples(
        ["abcdefgh"],
        ["abcdefg"],
        tokenizer=ByteTokenizer(),
        max_input_length=6,
        max_target_length=6,
    )

    assert prepared.noisy_texts == []
    assert prepared.clean_texts == []
    assert prepared.source_indices == []
    assert prepared.audit.excluded_pairs == 1
    assert prepared.audit.exclusions == [(0, "no_complete_aligned_word_group_fits_limit")]


def test_policy_retains_pair_at_exact_tokenizer_limit_and_split_ids_are_disjoint():
    tokenizer = ByteTokenizer()
    prepared = prepare_paired_examples(
        ["abc", "def", "ghi", "jkl"],
        ["abc", "def", "ghi", "jkl"],
        tokenizer=tokenizer,
        max_input_length=4,
        max_target_length=4,
    )

    assert prepared.noisy_texts == ["abc", "def", "ghi", "jkl"]
    assert prepared.audit.unchanged_pairs == 4

    pipeline = DataPipeline(seed=42)
    splits = pipeline.split_data(
        prepared.noisy_texts,
        prepared.clean_texts,
        prepared.noise_levels,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )
    partitioned = [value for split in splits.values() for value in split[0]]
    assert sorted(partitioned) == sorted(prepared.noisy_texts)
    assert len(partitioned) == len(set(partitioned))


def test_dataset_rejects_unprepared_pair_instead_of_silent_truncation():
    with pytest.raises(ValueError, match="max_input_length"):
        NormalizationDataset(
            ["abcdef"],
            ["abcdef"],
            ByteTokenizer(),
            max_input_length=4,
            max_target_length=4,
        )


def test_synthetic_pairs_are_prepared_before_their_noise_labels_are_returned():
    manifest = build_probability_manifest(
        [{"categories": ["abbreviation"]}],
        {"abbreviation": (0.0, 1.0)},
        resource_versions={"reviewed_lexicon": "test"},
    )
    pipeline = DataPipeline(seed=42, noise_manifest=manifest)
    pipeline.noise_gen.generate_pair = lambda _, pair_id, base_id: (
        "aang gandaaaa",
        "ang ganda",
        SyntheticPairLineage(
            pair_id=pair_id,
            base_sentence_id=base_id,
            manifest_id=manifest.manifest_id,
            resource_versions=manifest.resource_versions,
            applied_preserved_categories=[],
            applied_correctable_categories=["abbreviation"],
            seed_derivation=42,
        ),
    )

    noisy, clean, noise = pipeline.generate_synthetic_pairs(
        ["ang ganda"],
        tokenizer=ByteTokenizer(),
        max_input_length=5,
        max_target_length=4,
        target_size=1,
    )

    assert (noisy, clean) == (["aang"], ["ang"])
    assert noise == [pytest.approx(compute_noise_level("aang", "ang"))]


@pytest.mark.parametrize("entry_point", ["train.py", "evaluate.py", "run_experiment.py", "benchmark.py"])
def test_all_gold_entry_points_prepare_pairs_before_the_split(entry_point):
    source = (Path(__file__).parents[1] / "scripts" / entry_point).read_text(encoding="utf-8")

    assert "gold_prepared = prepare_paired_examples(" in source
    assert source.index("gold_prepared = prepare_paired_examples(") < source.index("pipeline.split_data")
