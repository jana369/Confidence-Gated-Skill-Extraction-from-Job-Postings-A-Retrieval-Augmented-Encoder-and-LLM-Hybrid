# import faiss, torch
# from transformers import AutoModel

# datastore_path = "datastore_100_skillspan/index_jobberta"

# index = faiss.read_index(datastore_path + "/index.trained")
# kernel = torch.load(datastore_path + "/kernel.pt")
# bias = torch.load(datastore_path + "/bias.pt")
# input_ids = torch.load(datastore_path + "/input_ids.pt")
# token_ids = torch.load(datastore_path + "/token_ids.pt")

# print("index.ntotal:", index.ntotal)
# print(
#     "input_ids shape:",
#     input_ids.shape if torch.is_tensor(input_ids) else len(input_ids),
# )

# print(
#     "token_ids shape:",
#     token_ids.shape if torch.is_tensor(token_ids) else len(token_ids),
# )

import json

with open("jobBERTa/nnose/src/analysis_features.jsonl") as f:
    records = [json.loads(l) for l in f]

# tokens where the base model's argmax disagrees with gold
model_wrong = [
    r
    for r in records
    if max(range(3), key=lambda i: r["Initial Predictions"][i]) != r["Gold Label"]
]

corrected = 0
unchanged = 0
made_worse = 0

for r in model_wrong:
    init_pred = max(range(3), key=lambda i: r["Initial Predictions"][i])
    final_pred = max(range(3), key=lambda i: r["Final prediction"][i])
    if final_pred == r["Gold Label"]:
        corrected += 1
    elif final_pred == init_pred:
        unchanged += 1
    else:
        made_worse += 1

print(
    f"corrected: {corrected}, unchanged: {unchanged}, made_worse: {made_worse}, total: {len(model_wrong)}"
)

corrected_tokens = []
for r in model_wrong:
    init_pred = max(range(3), key=lambda i: r["Initial Predictions"][i])
    final_pred = max(range(3), key=lambda i: r["Final prediction"][i])
    if final_pred == r["Gold Label"]:
        corrected_tokens.append(r)

for r in corrected_tokens:
    print(
        r["Current Subword"],
        "| gold:",
        r["Gold Label"],
        "| neighbor agreement:",
        r["Neighbor Agreement"],
        "| knn scores:",
        r["Aggregated kNN Scores"],
        "| init pred:",
        r["Initial Predictions"],
    )
