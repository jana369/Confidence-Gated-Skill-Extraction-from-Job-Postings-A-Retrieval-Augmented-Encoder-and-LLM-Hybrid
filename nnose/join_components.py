"""
join_components.py (v2 -- simplified)

Combines Component A's (JobBERTa+kNN) and Component B's (LLM) per-token
outputs into ONE feature table for Component C's gate training.

This version joins DIRECTLY on real keys now written by both sides --
no reconstruction, no guessing at batch/step math:
    Component A: "skillspan_row", "token_position"
    Component B: "idx",           llm_labels[i]["position"]

Usage:
    python join_components.py \
        --component-a analysis_features.jsonl \
        --component-b nvidia_dev_full.jsonl \
        --out gate_training_table.jsonl
"""

import argparse
import json
from collections import defaultdict


def load_component_a(path: str) -> dict:
    """Groups Component A's rows by (skillspan_row, token_position) ->
    the full token record, for direct O(1) lookup during the join."""
    by_key = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["skillspan_row"], row["token_position"])
            by_key[key] = row
    return by_key


def load_component_b(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run(component_a_path: str, component_b_path: str, out_path: str):
    print(f"Loading Component A from {component_a_path} ...")
    a_by_key = load_component_a(component_a_path)
    print(f"  {len(a_by_key)} token rows loaded")

    print(f"Loading Component B from {component_b_path} ...")
    b_records = load_component_b(component_b_path)
    print(f"  {len(b_records)} sentence rows loaded")

    joined = 0
    b_tokens_total = 0
    unmatched_b_tokens = 0

    out_f = open(out_path, "w", encoding="utf-8")

    matched_a_keys = set()
    for b_record in b_records:
        idx = b_record["idx"]
        llm_labels = b_record.get("llm_labels") or []

        for b_tok in llm_labels:
            b_tokens_total += 1
            position = b_tok["position"]
            key = (idx, position)

            a_tok = a_by_key.get(key)
            if a_tok is None:
                unmatched_b_tokens += 1
                continue

            matched_a_keys.add(key)

            joined_row = {
                "skillspan_row": idx,
                "token_position": position,
                "token": a_tok["Original Token"],
                "gold_label": a_tok["Gold Label"],
                # Component A features
                "a_prediction_probs": a_tok["Probabilities"]["Final"],
                "a_classifier_probs": a_tok["Probabilities"]["Initial"],
                "a_knn_probs": a_tok["Probabilities"]["kNN"],
                "a_neighbor_agreement": a_tok["Neighbor Agreement"],
                "a_mean_distance": (
                    sum(a_tok["Distances"]) / len(a_tok["Distances"])
                    if a_tok["Distances"]
                    else None
                ),
                "a_max_distance": (
                    max(a_tok["Distances"]) if a_tok["Distances"] else None
                ),
                "a_embedding_norm": a_tok["Embedding Norm"],
                # Component B features
                "b_prediction": b_tok.get("llm_label", "O"),
                "b_logprob": b_tok.get("logprob"),
            }
            out_f.write(json.dumps(joined_row) + "\n")
            joined += 1

    unmatched_a_only = len(a_by_key) - len(matched_a_keys)

    out_f.close()
    print(f"\nDone. Wrote {joined} joined token rows to {out_path}")
    print(
        f"Component B tokens with no Component A match: {unmatched_b_tokens} / {b_tokens_total}"
    )
    print(
        f"Component A tokens with no Component B match "
        f"(e.g. degenerate sentences skipped by B's guard): {unmatched_a_only}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-a", required=True)
    parser.add_argument("--component-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run(args.component_a, args.component_b, args.out)
