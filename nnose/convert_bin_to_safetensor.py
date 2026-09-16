from transformers import AutoModelForTokenClassification, AutoTokenizer

path = r"tmp-5e-5/green/jobberta/epoch_3"
model = AutoModelForTokenClassification.from_pretrained(path)

model.save_pretrained(
    r"tmp-5e-5/green/jobberta/epoch_3_safetensors", safe_serialization=True
)

tok = AutoTokenizer.from_pretrained(path)
tok.save_pretrained(
    r"tmp-5e-5/green/jobberta/epoch_3_safetensors"
)
