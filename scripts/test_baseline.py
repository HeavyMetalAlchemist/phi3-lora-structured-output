# scripts/test_baseline.py

import json
import random
import jsonlines
from pathlib import Path

# Import shared logic from baseline_eval
from baseline_eval import (
    extract_json,
    validate_schema,
    run_eval,
    VALID_EMOTIONS,
    VALID_INTENSITIES,
    VALID_FORMALITIES,
)


def mock_inference(text: str) -> str:
    """Returns random valid JSON — simulates model output."""
    return json.dumps({
        "emotion"   : random.choice(list(VALID_EMOTIONS)),
        "intensity" : random.choice(list(VALID_INTENSITIES)),
        "formality" : random.choice(list(VALID_FORMALITIES)),
        "actionable": random.choice([True, False]),
    })


def test_extract_json():
    # Clean JSON
    assert extract_json('{"emotion": "joy"}') == {"emotion": "joy"}
    # Wrapped in markdown
    assert extract_json('```json\n{"emotion": "joy"}\n```') == {"emotion": "joy"}
    # No JSON
    assert extract_json("no json here") is None
    print("extract_json: OK")


def test_validate_schema():
    # Valid
    assert validate_schema({
        "emotion": "joy", "intensity": "high",
        "formality": "informal", "actionable": True
    }) is True
    # Missing field
    assert validate_schema({
        "emotion": "joy", "intensity": "high",
        "formality": "informal"
    }) is False
    # Invalid value
    assert validate_schema({
        "emotion": "happiness", "intensity": "high",
        "formality": "informal", "actionable": True
    }) is False
    # actionable as string instead of bool
    assert validate_schema({
        "emotion": "joy", "intensity": "high",
        "formality": "informal", "actionable": "true"
    }) is False
    print("validate_schema: OK")


def test_run_eval():
    # Load local test set
    test_path = Path("../data/test.jsonl")
    if not test_path.exists():
        print("test_eval: SKIPPED — data/test.jsonl not found")
        return

    with jsonlines.open(test_path) as reader:
        test_examples = list(reader)[:10]  # only 10 for speed

    results, metrics = run_eval(
        test_examples,
        tokenizer=None,
        model=None,
        inference_fn=mock_inference,
    )

    assert len(results) == 10
    assert "schema_compliance"   in metrics
    assert "emotion_accuracy"    in metrics
    assert "intensity_accuracy"  in metrics
    assert "formality_accuracy"  in metrics
    assert "actionable_accuracy" in metrics
    assert "overall_accuracy"    in metrics

    # All metric values between 0 and 1
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k} out of range: {v}"

    print("run_eval: OK")
    print("Metrics on 10 mock examples:")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")


if __name__ == "__main__":
    test_extract_json()
    test_validate_schema()
    test_run_eval()
    print("\nAll tests passed.")