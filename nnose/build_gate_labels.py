"""
build_gate_labels.py

ONE function that takes the joined gate_training_table.jsonl and produces
the final gate-training dataset: derives Component A's discrete B/I/O
prediction via argmax, compares BOTH components against gold, and assigns
the training label per the paper's rule (Section 3.4):

    - only A correct  -> "trust_a"
    - only B correct  -> "trust_b"
    - both correct    -> "trust_a"   (A is cheaper, no LLM API cost bias toward the free option when it doesn't matter)
    - both wrong      -> excluded    (no informative signal on preference)

Label id mapping (confirmed from Component A's data): 0=B, 1=I, 2=O.
"""

import argparse
import json

ID_TO_LABEL = {0: "B-SKILL", 1: "I-SKILL", 2: "O"}


def build_gate_label(row: dict) -> dict:
    """
    Takes one joined row (from gate_training_table.jsonl) and returns it
    with two new fields added:
        "a_prediction": the discrete B-SKILL/I-SKILL/O label derived from
            a_prediction_probs via argmax (the paper's Final probability
            distribution -- classifier + kNN already interpolated).
        "gate_target": "trust_a" / "trust_b" / None (excluded -- both
            components wrong, no informative signal for the gate).

    Does not mutate the input dict; returns a new one.
    """
    gold_id = row["gold_label"]
    gold_label = ID_TO_LABEL.get(gold_id, "O")

    # Derive Component A's discrete prediction via argmax over its final
    # (classifier + kNN interpolated) probability distribution.
    a_probs = row["a_prediction_probs"]
    a_pred_id = a_probs.index(max(a_probs))
    a_prediction = ID_TO_LABEL[a_pred_id]

    b_prediction = row.get("b_prediction", "O")

    a_correct = a_prediction == gold_label
    b_correct = b_prediction == gold_label

    if a_correct and not b_correct:
        gate_target = "trust_a"
    elif b_correct and not a_correct:
        gate_target = "trust_b"
    elif a_correct and b_correct:
        gate_target = "trust_a"  # both correct -> default to cheaper option
    else:
        gate_target = None  # both wrong -> excluded from gate training

    out = dict(row)
    out["gold_label_str"] = gold_label
    out["a_prediction"] = a_prediction
    out["a_correct"] = a_correct
    out["b_correct"] = b_correct
    out["gate_target"] = gate_target
    return out


def run(input_path: str, output_path: str):
    n_total = 0
    n_trust_a = 0
    n_trust_b = 0
    n_excluded = 0
    n_both_correct = 0

    with open(input_path, "r", encoding="utf-8") as in_f, open(
        output_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            row = json.loads(line)
            labeled = build_gate_label(row)
            n_total += 1

            if labeled["gate_target"] == "trust_a":
                n_trust_a += 1
                if labeled["a_correct"] and labeled["b_correct"]:
                    n_both_correct += 1
            elif labeled["gate_target"] == "trust_b":
                n_trust_b += 1
            else:
                n_excluded += 1

            out_f.write(json.dumps(labeled) + "\n")

    print(f"Total tokens processed: {n_total}")
    print(f"  trust_both: {n_both_correct}")
    print(f"  trust_a: {n_trust_a}")
    print(f"  trust_b: {n_trust_b}")
    print(f"  excluded (both wrong): {n_excluded}")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in", dest="inp", required=True, help="gate_training_table.jsonl"
    )
    parser.add_argument(
        "--out", dest="out", required=True, help="Output path with gate_target labels"
    )
    args = parser.parse_args()

    run(args.inp, args.out)
