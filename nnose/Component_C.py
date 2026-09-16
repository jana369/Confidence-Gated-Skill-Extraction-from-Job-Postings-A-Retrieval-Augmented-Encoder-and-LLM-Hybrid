"""
component_c_gate.py

Component C -- THE GATE (the only trained component in the whole pipeline).

Per the project spec (Section 3.3-3.4):
    A small classifier (logistic regression or tiny MLP) that takes a
    handful of scalar features per token and outputs a routing decision:
    trust Component A (JobBERTa+kNN) or trust Component B (LLM).

Input:  gate_training_labeled.jsonl (already built tonight -- has every
        feature from A and B, plus the gate_target label per token).
Output: a trained, saved classifier + evaluation report.
"""

import joblib
import json
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# ============================================================
# STEP 1: Load and prepare the feature table
# ============================================================
FEATURE_COLUMNS = [
    "a_neighbor_agreement",
    "a_mean_distance",
    "a_max_distance",
    "a_max_prob_calibrated",
    "a_margin_calibrated",
    "a_pred_B-SKILL",
    "a_pred_I-SKILL",
    "a_pred_O",
]

MIN_PRECISION = 0.30
RANDOM_STATE = 42

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def load_gate_training_data(file_path: str) -> pd.DataFrame:
    
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            unique_id = f"{row['skillspan_row']}_{row['token_position']}"
            new_row = {"id": unique_id}
            new_row.update(row)
            rows.append(new_row)
    df = pd.DataFrame(rows)
    return df


def filter_excluded_tokens(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["gate_target"].notna()].copy()


def derive_a_features(df: pd.DataFrame) -> pd.DataFrame:
    df["a_max_prob"] = df["a_prediction_probs"].apply(max)
    df["a_margin"] = df["a_prediction_probs"].apply(
        lambda probs: sorted(probs, reverse=True)[0] - sorted(probs, reverse=True)[1]
    )
    for label in ["B-SKILL", "I-SKILL", "O"]:
        df[f"a_pred_{label}"] = (df["a_prediction"] == label).astype(int)
    return df


def fit_calibrator(df_train: pd.DataFrame, raw_col: str = "a_max_prob"):
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(df_train[raw_col].values, df_train["a_correct"].astype(int).values)
    return calibrator


def apply_calibration(
    df: pd.DataFrame, calibrator, raw_col: str = "a_max_prob"
) -> pd.DataFrame:
    df = df.copy()
    df["a_max_prob_calibrated"] = calibrator.predict(df[raw_col].values)

    top1_top2 = df["a_prediction_probs"].apply(lambda p: sorted(p, reverse=True)[:2])
    top1 = np.array([t[0] for t in top1_top2])
    top2 = np.array([t[1] for t in top1_top2])
    cal_top1 = calibrator.predict(top1)
    cal_top2 = calibrator.predict(top2)
    df["a_margin_calibrated"] = cal_top1 - cal_top2

    return df


# ============================================================
# STEP 3: Train / test split
# ============================================================


def extract_feature_vector(df: pd.DataFrame):
    """
    Builds the working feature table. Keeps a_correct and
    a_prediction_probs alongside the base features (rather than
    restricting to FEATURE_COLUMNS immediately) because downstream
    calibration fitting needs a_correct as its target and
    a_prediction_probs to recompute margins -- both are
    Component-A-derived and available before B is consulted.
    FEATURE_COLUMNS' *_calibrated columns do not exist until
    apply_calibration() has run.
    """
    required = ["a_neighbor_agreement", "a_mean_distance", "a_max_distance"]
    X = df[df[required].notna().all(axis=1)].copy()
    y = X["gate_target"].tolist()
    return X, y


# ============================================================
# STEP 4: Train the gate classifier
# ============================================================


def select_threshold(precisions, recalls, thresholds, min_precision: float):
    valid = precisions[:-1] >= min_precision
    if not valid.any():
        return None, None, None, None
    candidate_recalls = np.where(valid, recalls[:-1], -1)
    idx = np.argmax(candidate_recalls)
    return thresholds[idx], precisions[idx], recalls[idx], idx


def cross_validated_threshold(X: pd.DataFrame, y: list, n_splits: int = 5):
    y_arr = np.array(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_thresholds = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_arr)):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Fit calibration fresh per fold, on this fold's train portion only.
        fold_calibrator = fit_calibrator(X_train_raw)
        X_train = apply_calibration(X_train_raw, fold_calibrator)[FEATURE_COLUMNS]
        X_test = apply_calibration(X_test_raw, fold_calibrator)[FEATURE_COLUMNS]

        clf = RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_estimators=300
        )
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        trust_b_idx = list(clf.classes_).index("trust_b")
        scores = proba[:, trust_b_idx]
        y_test_binary = (y_test == "trust_b").astype(int)

        precisions, recalls, thresholds = precision_recall_curve(y_test_binary, scores)
        t, p, r, _ = select_threshold(precisions, recalls, thresholds, MIN_PRECISION)
        fold_thresholds.append(t)
        print(
            f"  Fold {fold_idx + 1}: threshold={t:.4f}  precision={p:.3f}  recall={r:.3f}"
        )

    fold_thresholds = [t for t in fold_thresholds if t is not None]
    print(
        f"\n  Threshold across folds: mean={np.mean(fold_thresholds):.4f}  "
        f"std={np.std(fold_thresholds):.4f}  "
        f"min={np.min(fold_thresholds):.4f}  max={np.max(fold_thresholds):.4f}"
    )
    return fold_thresholds


# ============================================================
# STEP 5: Evaluate the gate
# ============================================================


def train_final_gate(X: pd.DataFrame, y: list, min_precision: float = MIN_PRECISION):
    X_train_raw, X_holdout_raw, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Fit the calibrator on the TRAIN portion only, then apply to both
    # train (for consistency) and holdout -- never fit on holdout/test.
    calibrator = fit_calibrator(X_train_raw)
    X_train = apply_calibration(X_train_raw, calibrator)
    X_holdout = apply_calibration(X_holdout_raw, calibrator)

    clf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_estimators=300
    )
    clf.fit(X_train[FEATURE_COLUMNS], y_train)

    print("=== Baseline: fixed 0.5 threshold (default predict()) ===")
    y_pred_default = clf.predict(X_holdout[FEATURE_COLUMNS])
    print(classification_report(y_holdout, y_pred_default, digits=3))

    proba = clf.predict_proba(X_holdout[FEATURE_COLUMNS])
    trust_b_idx = list(clf.classes_).index("trust_b")
    trust_b_scores = proba[:, trust_b_idx]
    y_holdout_binary = np.array([1 if v == "trust_b" else 0 for v in y_holdout])

    precisions, recalls, thresholds = precision_recall_curve(
        y_holdout_binary, trust_b_scores
    )
    threshold, precision_at_t, recall_at_t, _ = select_threshold(
        precisions, recalls, thresholds, min_precision
    )

    print(f"\n=== Selected threshold: max recall s.t. precision >= {min_precision} ===")
    print(
        f"threshold={threshold:.4f}  precision={precision_at_t:.3f}  recall={recall_at_t:.3f}"
    )

    y_pred_final = np.where(trust_b_scores >= threshold, "trust_b", "trust_a")
    print("\n=== Full report at selected threshold ===")
    print(classification_report(y_holdout, y_pred_final, digits=3))

    return clf, threshold, X_holdout, y_holdout, clf.feature_importances_, calibrator


# ============================================================
# STEP 6: Save / load the trained gate
# ============================================================


def save_gate(model, threshold: float, calibrator, file_path: str) -> None:
    """Persists the trained classifier AND the fitted calibrator, so
    both can be reused for inference on the TEST split without
    retraining or re-fitting calibration -- re-fitting on test data
    would leak test-set accuracy into the calibration mapping itself."""
    joblib.dump(
        {"model": model, "threshold": threshold, "calibrator": calibrator}, file_path
    )
    print(f"Saved gate (model + threshold={threshold:.4f} + calibrator) to {file_path}")


def load_gate(file_path: str):
    bundle = joblib.load(file_path)
    return bundle["model"], bundle["threshold"], bundle["calibrator"]


def calibration_check(df: pd.DataFrame, n_bins: int = 10):
    """
    Checks whether a_max_prob (RAW, pre-calibration) is calibrated.
    Kept on the raw score deliberately -- this is the diagnostic that
    motivated adding fit_calibrator/apply_calibration in the first
    place, so it should keep measuring the same raw quantity it always
    did, as a before/after reference point.
    """
    y_true = df["a_correct"].astype(int).values
    y_prob = df["a_max_prob"].values

    bin_accuracy, bin_confidence = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_prob)

    print("=== Component A calibration check (RAW, pre-calibration) ===")
    print(f"Brier score: {brier:.4f}  (0 = perfect, 0.25 = uninformative)\n")
    for conf, acc in zip(bin_confidence, bin_accuracy):
        gap = conf - acc
        flag = (
            "  <- overconfident"
            if gap > 0.05
            else ("  <- underconfident" if gap < -0.05 else "")
        )
        print(f"predicted={conf:.3f}  actual={acc:.3f}  gap={gap:.3f}{flag}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(bin_confidence, bin_accuracy, marker="o", label="Component A (raw)")
    ax.set_xlabel("Mean predicted confidence (a_max_prob, raw)")
    ax.set_ylabel("Actual fraction correct")
    ax.set_title(f"Calibration curve, RAW (Brier score: {brier:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("calibration_curve_raw.png", dpi=150)
    print("Calibration plot saved to calibration_curve_raw.png")

    return brier


def calibration_check_after(df: pd.DataFrame, calibrator, n_bins: int = 10):
    """
    Same diagnostic as calibration_check(), but on the CALIBRATED score
    (a_max_prob_calibrated), computed with the same held-out-safe
    calibrator used everywhere else. This is the number that should
    have moved toward the diagonal if calibration is working -- compare
    directly against calibration_check()'s Brier score on the same rows.
    """
    df = apply_calibration(df, calibrator)
    y_true = df["a_correct"].astype(int).values
    y_prob = df["a_max_prob_calibrated"].values

    bin_accuracy, bin_confidence = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_prob)

    print("=== Component A calibration check (AFTER calibration) ===")
    print(f"Brier score: {brier:.4f}  (0 = perfect, 0.25 = uninformative)\n")
    for conf, acc in zip(bin_confidence, bin_accuracy):
        gap = conf - acc
        flag = (
            "  <- overconfident"
            if gap > 0.05
            else ("  <- underconfident" if gap < -0.05 else "")
        )
        print(f"predicted={conf:.3f}  actual={acc:.3f}  gap={gap:.3f}{flag}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(
        bin_confidence,
        bin_accuracy,
        marker="o",
        color="green",
        label="Component A (calibrated)",
    )
    ax.set_xlabel("Mean predicted confidence (calibrated)")
    ax.set_ylabel("Actual fraction correct")
    ax.set_title(f"Calibration curve, AFTER (Brier score: {brier:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("calibration_curve_after.png", dpi=150)
    print("Calibration plot saved to calibration_curve_after.png")

    return brier


def plot_pr_curve(
    model, X_holdout, y_holdout, chosen_threshold=None, min_precision=None
):
    proba = model.predict_proba(X_holdout[FEATURE_COLUMNS])
    trust_b_idx = list(model.classes_).index("trust_b")
    scores = proba[:, trust_b_idx]
    y_binary = np.array([1 if v == "trust_b" else 0 for v in y_holdout])

    precisions, recalls, thresholds = precision_recall_curve(y_binary, scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, label="PR curve (trust_b)")
    if min_precision is not None:
        ax.axhline(
            min_precision,
            linestyle="--",
            color="red",
            label=f"MIN_PRECISION={min_precision}",
        )
    if chosen_threshold is not None:
        idx = np.argmin(np.abs(thresholds - chosen_threshold))
        ax.scatter(
            [recalls[idx]],
            [precisions[idx]],
            color="black",
            zorder=5,
            s=80,
            label=f"Chosen (P={precisions[idx]:.2f}, R={recalls[idx]:.2f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve: trust_b (escalate to LLM)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("pr_curve.png", dpi=150)
    print("PR curve saved to pr_curve.png")

    valid = precisions[:-1] >= (min_precision or 0.30)
    if valid.any():
        print(
            f"\nAt precision >= {min_precision or 0.30}: best achievable recall = {recalls[:-1][valid].max():.3f}"
        )
    else:
        print(f"\nNo threshold achieves precision >= {min_precision or 0.30} at all.")

    return precisions, recalls, thresholds


def oracle_analysis(test_data_path: str) -> dict:
    """
    Oracle: for each token, if A is correct, use A; elif B is correct, use B;
    else the token is unfixable by any routing choice (both wrong). This is
    the upper bound on what ANY gate -- however well-trained -- could achieve
    by choosing between A and B, since it always picks the correct one
    whenever at least one of them is correct.

    IMPORTANT: unlike the trained Gate's own evaluation, this runs on the
    FULL test set with NO filter_excluded_tokens() call. The whole point
    of an oracle is to show the true ceiling INCLUDING tokens neither
    component can solve -- filtering those out first would make the
    oracle trivially perfect (1.000) by construction, not meaningfully
    informative. Compare this oracle F1 against Component A alone /
    Component B alone / the trained Gate's hybrid F1, all computed on
    this SAME unfiltered set, for a fair four-way comparison.

    Interpretation:
        oracle F1 ~= Component A alone F1  -> A and B mostly agree/fail
            together; little headroom for any gate to exploit.
        oracle F1 >> Component A alone F1  -> A and B are meaningfully
            complementary; a gap between oracle and the trained Gate's
            actual F1 means the Gate is leaving real headroom on the table.
        oracle F1 < 1.000                  -> the "both wrong" tokens are
            real and present; this is the honest ceiling, not a
            filtering artifact.
    """
    df = load_gate_training_data(test_data_path)
    # NOTE: deliberately NOT calling filter_excluded_tokens() here --
    # see docstring above.

    gold = df["gold_label_str"].tolist()
    a_pred = df["a_prediction"].tolist()
    b_pred = df["b_prediction"].tolist()
    a_correct = df["a_correct"].tolist()
    b_correct = df["b_correct"].tolist()

    oracle_pred = [
        a_p if a_c else (b_p if b_c else a_p)  # both-wrong: A's guess, arbitrary
        for a_p, b_p, a_c, b_c in zip(a_pred, b_pred, a_correct, b_correct)
    ]

    report = classification_report(gold, oracle_pred, digits=3, output_dict=True)
    print("=== ORACLE (upper bound: always picks whichever of A/B is correct) ===")
    print(
        "(unfiltered -- includes both-wrong tokens, so this is a real ceiling, not 1.000 by construction)"
    )
    print(classification_report(gold, oracle_pred, digits=3))
    print(f"\nOracle macro-F1: {report['macro avg']['f1-score']:.3f}")

    return report


def error_overlap_analysis(test_data_path: str) -> dict:
    """
    Classifies every test token into one of four buckets based on
    whether A and B were each correct, and reports counts/percentages.
    Directly answers: do A and B make genuinely different errors?

    IMPORTANT: like oracle_analysis(), this runs on the FULL, unfiltered
    test set -- filter_excluded_tokens() would remove every possible
    "both wrong" row before it could be counted, making that bucket
    read 0% regardless of the true rate. This function's whole purpose
    is to measure that rate honestly.

    The single most important number here (per supervisor's framing):
    how many tokens fall into "B correct, A wrong" -- these are exactly
    the tokens where the Gate's job is to notice A is unreliable and
    escalate to B. A small count here means little headroom exists for
    ANY gate; a large count means there's real complementarity to exploit.
    The "both wrong" bucket is equally important to report honestly: it's
    the portion of errors NO routing decision can ever fix.
    """
    df = load_gate_training_data(test_data_path)
    # NOTE: deliberately NOT calling filter_excluded_tokens() here --
    # see docstring above.

    a_correct = df["a_correct"]
    b_correct = df["b_correct"]

    both_correct = (a_correct & b_correct).sum()
    a_only_correct = (a_correct & ~b_correct).sum()
    b_only_correct = (~a_correct & b_correct).sum()
    both_wrong = (~a_correct & ~b_correct).sum()
    total = len(df)

    print("=== ERROR OVERLAP / COMPLEMENTARITY (full test set, unfiltered) ===")
    print(f"Total tokens: {total}\n")
    print(f"Both correct:        {both_correct:6d}  ({both_correct/total:.1%})")
    print(f"A correct, B wrong:  {a_only_correct:6d}  ({a_only_correct/total:.1%})")
    print(
        f"B correct, A wrong:  {b_only_correct:6d}  ({b_only_correct/total:.1%})  <-- headroom for the Gate"
    )
    print(
        f"Both wrong:          {both_wrong:6d}  ({both_wrong/total:.1%})  <-- unfixable by any routing"
    )

    return {
        "both_correct": int(both_correct),
        "a_only_correct": int(a_only_correct),
        "b_only_correct": int(b_only_correct),
        "both_wrong": int(both_wrong),
        "total": int(total),
    }


def gate_routing_analysis(
    model, threshold: float, calibrator, test_data_path: str
) -> dict:
    """
    Breaks down the Gate's actual routing decisions on Gate-test against
    the four error-overlap buckets (both_correct / a_only_correct /
    b_only_correct / both_wrong), answering: given the type of token,
    does the Gate route it the way it should?

    IMPORTANT: like oracle_analysis() and error_overlap_analysis(), this
    runs on the FULL, unfiltered test set -- no filter_excluded_tokens()
    call. All three functions report on the same four buckets over the
    same population, and are meant to be read side by side; filtering
    here alone would silently drop every both_wrong token before the
    Gate's routing on them could be measured, and would leave this
    function's bucket totals inconsistent with the other two.

    The single most important number here: among b_only_correct tokens
    (the ones where routing to B is the ONLY way to get this token right),
    what fraction did the Gate actually send to B? A low number here means
    the Gate itself is failing to exploit the complementarity that exists
    (a Gate-quality problem); if this number is already high, the small
    oracle-vs-hybrid gap is mostly just the inherent ceiling, not a fixable
    Gate shortcoming. The both_wrong bucket is also worth reporting here:
    it shows how the Gate routes tokens that are unfixable either way
    (ideally to A, the cheaper option, since routing them to B wins nothing).
    """
    df = load_gate_training_data(test_data_path)
    # NOTE: deliberately NOT calling filter_excluded_tokens() here --
    # see docstring above.
    df = derive_a_features(df)
    df = apply_calibration(df, calibrator)

    X = df[df[FEATURE_COLUMNS].notna().all(axis=1)].copy()

    proba = model.predict_proba(X[FEATURE_COLUMNS])
    trust_b_idx = list(model.classes_).index("trust_b")
    trust_b_scores = proba[:, trust_b_idx]
    gate_decision = np.where(trust_b_scores >= threshold, "trust_b", "trust_a")
    X["gate_decision"] = gate_decision

    a_correct = X["a_correct"]
    b_correct = X["b_correct"]

    buckets = {
        "both_correct": a_correct & b_correct,
        "a_only_correct": a_correct & ~b_correct,
        "b_only_correct": ~a_correct & b_correct,
        "both_wrong": ~a_correct & ~b_correct,
    }

    print("=== GATE ROUTING ANALYSIS (Gate-test) ===")
    print(
        f"Overall: routed to A = {(gate_decision == 'trust_a').sum()}  "
        f"routed to B = {(gate_decision == 'trust_b').sum()}\n"
    )

    results = {}
    for bucket_name, mask in buckets.items():
        bucket_df = X[mask]
        n = len(bucket_df)
        if n == 0:
            print(f"{bucket_name}: 0 tokens, skipping")
            continue
        routed_a = (bucket_df["gate_decision"] == "trust_a").sum()
        routed_b = (bucket_df["gate_decision"] == "trust_b").sum()
        pct_a = routed_a / n
        pct_b = routed_b / n

        results[bucket_name] = {
            "n": int(n),
            "routed_to_a": int(routed_a),
            "routed_to_b": int(routed_b),
            "pct_routed_to_a": float(pct_a),
            "pct_routed_to_b": float(pct_b),
        }

        print(f"{bucket_name}  (n={n}):")
        print(f"    routed to A: {routed_a:5d}  ({pct_a:.1%})")
        print(f"    routed to B: {routed_b:5d}  ({pct_b:.1%})")

    if "b_only_correct" in results:
        headline = results["b_only_correct"]["pct_routed_to_b"]
        print(
            f"\n*** KEY NUMBER: among B-only-correct tokens, the Gate routed "
            f"{headline:.1%} to B (the only correct choice for these tokens). ***"
        )
        if headline < 0.5:
            print(
                "    Less than half -- the Gate is missing most of the tokens "
                "where escalating to B is the ONLY way to get them right."
            )
        else:
            print(
                "    More than half -- the Gate captures most of the available "
                "B-only-correct opportunity; the small oracle-vs-hybrid gap is "
                "likely closer to the structural ceiling than a Gate defect."
            )

    if "a_only_correct" in results:
        false_escalation = results["a_only_correct"]["pct_routed_to_b"]
        print(
            f"\n*** For contrast: among A-only-correct tokens, the Gate "
            f"incorrectly routed {false_escalation:.1%} to B (a token this "
            f"harms). ***"
        )

    return results


# ============================================================
# STEP 7: Apply the trained gate at inference time
# ============================================================


def evaluate_on_test_split(model, threshold: float, calibrator, test_data_path: str):
    df = load_gate_training_data(test_data_path)
    df = filter_excluded_tokens(df)
    df = derive_a_features(df)
    df = apply_calibration(df, calibrator)

    X = df[df[FEATURE_COLUMNS].notna().all(axis=1)].copy()
    y_true = X["gate_target"].tolist()
    X_features = X[FEATURE_COLUMNS]

    print(f"Total usable TEST rows: {len(X_features)}")
    print(f"Class balance (TEST): {pd.Series(y_true).value_counts().to_dict()}\n")

    proba = model.predict_proba(X_features)
    trust_b_idx = list(model.classes_).index("trust_b")
    trust_b_scores = proba[:, trust_b_idx]

    y_pred = np.where(trust_b_scores >= threshold, "trust_b", "trust_a")

    print(
        f"=== FINAL TEST SET RESULTS (fixed threshold={threshold:.4f}, no retuning) ==="
    )
    print(classification_report(y_true, y_pred, digits=3))

    return classification_report(y_true, y_pred, digits=3, output_dict=True)


def evaluate_end_to_end_pipeline(
    model, threshold: float, calibrator, test_data_path: str
) -> dict:
    df = load_gate_training_data(test_data_path)
    df = filter_excluded_tokens(df)
    df = derive_a_features(df)
    df = apply_calibration(df, calibrator)

    X = df[df[FEATURE_COLUMNS].notna().all(axis=1)].copy()

    proba = model.predict_proba(X[FEATURE_COLUMNS])
    trust_b_idx = list(model.classes_).index("trust_b")
    trust_b_scores = proba[:, trust_b_idx]
    gate_decision = np.where(trust_b_scores >= threshold, "trust_b", "trust_a")

    gold = X["gold_label_str"].tolist()
    a_only = X["a_prediction"].tolist()
    b_only = X["b_prediction"].tolist()

    hybrid = [
        a_pred if decision == "trust_a" else b_pred
        for a_pred, b_pred, decision in zip(a_only, b_only, gate_decision)
    ]

    def score(predictions, name):
        report = classification_report(gold, predictions, digits=3, output_dict=True)
        print(f"\n=== {name} ===")
        print(classification_report(gold, predictions, digits=3))
        return report

    print(f"Evaluating on {len(X)} test tokens\n")

    report_a = score(a_only, "Component A alone")
    report_b = score(b_only, "Component B alone")
    report_hybrid = score(hybrid, "HYBRID (gate-routed) -- the real pipeline")

    print("\n=== HEADLINE COMPARISON (macro avg F1) ===")
    print(f"Component A alone : {report_a['macro avg']['f1-score']:.3f}")
    print(f"Component B alone : {report_b['macro avg']['f1-score']:.3f}")
    print(f"Hybrid (gated)    : {report_hybrid['macro avg']['f1-score']:.3f}")

    return {
        "component_a": report_a,
        "component_b": report_b,
        "hybrid": report_hybrid,
    }


# ============================================================
# Orchestration
# ============================================================


def main():
    DATA_PATH = "D:\\conference\\jobBERTa\\nnose\\gate_training_labeled.jsonl"
    GATE_SAVE_PATH = "D:\\conference\\jobBERTa\\nnose\\trained_gate.joblib"

    df = load_gate_training_data(DATA_PATH)
    df = filter_excluded_tokens(df)
    df = derive_a_features(df)
    X, y = extract_feature_vector(df)

    print(f"Total usable dev rows: {len(X)}")
    print(f"Class balance: {pd.Series(y).value_counts().to_dict()}\n")

    print("############################################")
    print("# Step 1: 5-fold CV threshold stability check")
    print("############################################")
    fold_thresholds = cross_validated_threshold(X, y, n_splits=5)
    print(f"\nFold thresholds: {fold_thresholds}")
    print("\n############################################")
    print("# Step 2: Final Gate + threshold (single dev holdout)")
    print("############################################")
    clf, final_threshold, X_holdout, y_holdout, feature_importance, calibrator = (
        train_final_gate(X, y)
    )
    print(f"Feature importances: {dict(zip(FEATURE_COLUMNS, feature_importance))}")

    print("\n############################################")
    print("# Step 2b: Calibration check (before/after) + PR curve")
    print("############################################")
    calibration_check(df)  # raw, whole-dev, for reference
    calibration_check_after(X_holdout.copy(), calibrator)  # calibrated, holdout only
    plot_pr_curve(
        clf,
        X_holdout,
        y_holdout,
        chosen_threshold=final_threshold,
        min_precision=MIN_PRECISION,
    )

    print(f"\nFinal deployed threshold: {final_threshold:.4f}")
    save_gate(clf, final_threshold, calibrator, GATE_SAVE_PATH)

    print("\n############################################")
    print("# [DEV, TRAIN-CONTAMINATED -- SANITY CHECK ONLY]")
    print("# clf was fit on 80% of this same DATA_PATH, so this")
    print("# number is NOT a valid estimate of held-out performance.")
    print("# Do NOT report this in the paper -- see the TEST-split")
    print("# call below for the real number.")
    print("############################################")
    print(evaluate_end_to_end_pipeline(clf, final_threshold, calibrator, DATA_PATH))
    print(
        "\nDev-set training complete. To get final paper-reportable numbers, "
        "run evaluate_on_test_split() with the TEST split's gate_training_labeled.jsonl."
    )


if __name__ == "__main__":
    main()
    print("\n############################################")
    print("# [TEST, HELD-OUT -- THIS IS THE REAL NUMBER]")
    print("############################################")
    model, threshold, calibrator = load_gate("trained_gate.joblib")
    evaluate_end_to_end_pipeline(
        model, threshold, calibrator, "gate_labeled_TEST.jsonl"
    )
    print(error_overlap_analysis("gate_labeled_TEST.jsonl"))
    print(oracle_analysis("gate_labeled_TEST.jsonl"))
    print(
        gate_routing_analysis(model, threshold, calibrator, "gate_labeled_TEST.jsonl")
    )
