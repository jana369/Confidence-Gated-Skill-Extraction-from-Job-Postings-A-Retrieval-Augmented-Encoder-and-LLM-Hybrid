# Confidence-Gated Skill Extraction from Job Postings

Code for the paper *"Confidence-Gated Skill Extraction from Job Postings: A Retrieval-Augmented Encoder and LLM Hybrid"*.

A hybrid skill-extraction pipeline combining:
- **Component A**: a retrieval-augmented fine-tuned encoder (JobBERTa + FAISS kNN), based on [NNOSE]([link-to-nnose-paper-or-repo](https://github.com/jjzha/nnose/tree/main#repository-for-nnose))
- **Component B**: a few-shot LLM-based extractor (Nemotron-49B via OpenRouter/NVIDIA)
- **Component C**: a lightweight trained Gate that routes each token between A and B

## Repository Structure

nnose/
├── src/ # Component A: training, inference, retrieval
├── LLM_Experiment/ # Component B: prompts, LLM client, alignment/evaluation
├── Component_C.py # The Gate: features, calibration, routing
├── build_gate_labels.py # Gate training-label construction
├── analyze_gate.py # Gate diagnostics / routing analysis
├── join_components.py # Combines A + B + Gate into final hybrid predictions
├── run_scripts/ # Shell scripts for training/inference
└── environment.yml


## Setup

```bash
conda env create -f nnose/environment.yml
conda activate <env-name>
```

Copy `.env.example` to `.env` and add your own API keys:
```bash
cp nnose/LLM_Experiment/.env.example nnose/LLM_Experiment/.env
```

GROQ_API_KEY=your_key_here
NVIDIA_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here


## Data

This project uses three publicly available datasets, **not redistributed here**:
- **SkillSpan**: [link to original source / cite zhang2022skillspan]
- **Sayfullina**: [link to original source]
- **Green**: [link to original source]

Download and place raw files under `nnose/data/{skillspan,sayfullina,green}/`. Use `nnose/data/skillspan/conll_to_json.py` (or the equivalent in `src/utils/`) to convert CoNLL format to the JSON format expected by the pipeline.

## Reproducing Results

- **Component A (NNOSE reproduction, Table 1)**: `run_scripts/run_trainer.sh`, `run_scripts/run_test.sh`
- **Component B model selection (Table 2)**: `LLM_Experiment/run_skillspan.py` with the model list in that script; evaluated via `LLM_Experiment/evaluate.py`
- **Gate training and calibration (Section on Gate)**: `build_gate_labels.py` → `Component_C.py`
- **Final hybrid evaluation (Table 4, Oracle/Error-Overlap analysis)**: `join_components.py`, `analyze_gate.py`

## Dependencies

Component A builds on [NNOSE](original-nnose-repo-link) (unmodified except for compatibility patches noted below). Clone it separately:
```bash
git clone <original-nnose-repo-url> jobBERTa
```

**Compatibility patches applied** (not in the original NNOSE code): deprecated Hugging Face arguments, Windows file-locking behavior, `faiss-gpu` API changes for a current PyTorch/FAISS stack.

## Trained Model / Large Files

The trained Gate (`trained_gate.joblib`) and datastore embeddings are not included in this repository due to size. [Available at: Hugging Face Hub / Zenodo link, or note "available on request"].

## Citation

```bibtex
@inproceedings{yourbibkey2026,
  title={Confidence-Gated Skill Extraction from Job Postings: A Retrieval-Augmented Encoder and LLM Hybrid},
  author={...},
  booktitle={...},
  year={2026}
}
```

## License

MIT (see `LICENSE`). Note: `jobBERTa/` (NNOSE, cloned separately per above) retains its own original license.
