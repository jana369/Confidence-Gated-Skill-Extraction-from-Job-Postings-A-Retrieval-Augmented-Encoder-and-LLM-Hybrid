# analyze_gate.py
import pandas as pd
import json
import matplotlib.pyplot as plt

ID_TO_LABEL = {0: "B-SKILL", 1: "I-SKILL", 2: "O"}
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

def load_data(file_path):
    """Load the labeled JSONL into a pandas DataFrame for easy plotting."""
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            
            # Create the id string
            unique_id = f"{row['skillspan_row']}_{row['token_position']}"
            
            # Create a new dictionary with 'id' as the very first key
            new_row = {'id': unique_id}
            new_row.update(row) # Add all the rest of the original data
            
            rows.append(new_row)

    df = pd.DataFrame(rows)  
    # --- Pre-calculate useful columns for your charts ---
    # Extract A's max confidence
    df["a_max_prob"] = df["a_prediction_probs"].apply(max)

    # Calculate margin (top1 - top2)
    df["a_margin"] = df["a_prediction_probs"].apply(
        lambda p: sorted(p, reverse=True)[0] - sorted(p, reverse=True)[1]
    )

    # Handle excluded tokens if you want to filter them out for certain plots
    df_filtered = df[df["gate_target"].notna()].copy()

    return df, df_filtered


def plot_gold_distribution(df):
    """Chart 1: Gold Label Distribution"""
    # TODO: You write the plotting logic here using df['gold_label_str']
    counts = df['gold_label'].value_counts()
    string_labels = [ID_TO_LABEL[i] for i in counts.index]
    plt.pie(counts.values, labels=string_labels, autopct='%1.1f%%', colors=["#4e0f63aa","#e4c7ff","#684b7ceb"])
    plt.title("Gold Label Distribution")
    plt.show()
    return 

def plot_Gate_Target_Distribution(df):
    """Chart 2: Gate Target Distribution"""
    counts = df["gate_target"].value_counts(dropna=False)
    counts.index = counts.index.fillna("excluded")

    plt.bar(
        x=counts.index,
        height=counts.values,
        width=0.8,
        color=["#4e0f63aa", "#e4c7ff", "#684b7ceb"],
    )
    plt.xlabel("Gate Target")
    plt.ylabel("Number of Tokens")
    plt.title("Gate Target Distribution")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 500, str(v), ha='center', fontweight='bold') 
    plt.show()
    return


def plot_gate_target_by_gold_normalized(df):
    """Chart 3b: Gate Target by Gold Label, as % within each gold label"""
    gate_target_clean = df["gate_target"].fillna("excluded")
    crosstab = pd.crosstab(df["gold_label_str"], gate_target_clean)
    crosstab = crosstab[["trust_a", "trust_b", "excluded"]]

    # Normalize each row (gold label) to sum to 100%
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    ax = crosstab_pct.plot(
        kind="bar",
        stacked=True,  # stacked, so each bar reaches exactly 100%
        figsize=(8, 6),
        color=["#4e0f63aa", "#e4c7ff", "#684b7ceb"],
        edgecolor="black",
    )

    # Annotate each segment with its percentage, since stacked bars can
    # be hard to read precisely by eye alone
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=9)

    plt.title("Gate Target by Gold Label (% within each label)")
    plt.xlabel("Gold Label")
    plt.ylabel("Percentage of Tokens")
    plt.xticks(rotation=10)
    plt.legend(title="Gate Target", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def build_true_comparison_category(row):
    """
    Categorizes the actual correctness relationship between Components A
    and B independently of the gate's labeling convention. The gate assigns
    trust_a when both components are correct, but this analysis treats such
    cases as ties because neither component is more accurate. A-only and
    B-only correct cases are therefore counted separately.
    """

    if row["a_correct"] and row["b_correct"]:
        return "both_correct"
    elif row["a_correct"] and not row["b_correct"]:
        return "a_only_correct"
    elif row["b_correct"] and not row["a_correct"]:
        return "b_only_correct"
    else:
        return "both_wrong"


def plot_true_comparison_by_gold(df):
    """Chart: TRUE model comparison by gold label -- distinguishes ties
    (both correct) from genuine wins, unlike the gate_target chart which
    collapses ties into trust_a and can visually overstate A's advantage.
    """
    df = df.copy()
    df["true_comparison"] = df.apply(build_true_comparison_category, axis=1)

    crosstab = pd.crosstab(df["gold_label_str"], df["true_comparison"])
    crosstab = crosstab[
        ["both_correct", "a_only_correct", "b_only_correct", "both_wrong"]
    ]

    crosstab.plot(
        kind="bar",
        stacked=False,
        figsize=(9, 6),
        color=["#c7c7c7", "#4e0f63aa", "#e4c7ff", "#684b7ceb"],
        edgecolor="black",
    )

    plt.title(
        "True Model Comparison by Gold Label\n(distinguishing ties from genuine wins)"
    )
    plt.xlabel("Gold Label")
    plt.ylabel("Number of Tokens")
    plt.xticks(rotation=10)
    plt.legend(title="Outcome")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    INPUT_PATH = "D:/conference/jobBERTa/nnose/gate_training_labeled.jsonl"

    df, df_filtered = load_data(INPUT_PATH)
    print(f"check unique ids: ",df['id'].nunique())
    print(f"check duplicates: ",df['id'].duplicated().sum())
    print(f"Loaded {len(df)} total rows.")
    print(f"Excluded (both wrong): {len(df) - len(df_filtered)}")
    print(f"the first 5 rows: ",df.head())
    print(f"columns of dataset: ",df.columns.tolist())
    # Call your plotting functions here
    plot_gold_distribution(df)
    plot_Gate_Target_Distribution(df)
    plot_gate_target_by_gold_normalized(df)
    plot_true_comparison_by_gold(df)
