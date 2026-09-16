# Confidence-Gated Skill Extraction from Job Postings

Code for the paper *"Confidence-Gated Skill Extraction from Job Postings: A Retrieval-Augmented Encoder and LLM Hybrid"*.

A hybrid skill-extraction pipeline combining:
- **Component A**: a retrieval-augmented fine-tuned encoder (JobBERTa + FAISS kNN), based on [NNOSE]([link-to-nnose-paper-or-repo](https://github.com/jjzha/nnose/tree/main#repository-for-nnose))
- **Component B**: a few-shot LLM-based extractor (Nemotron-49B via OpenRouter/NVIDIA)
- **Component C**: a lightweight trained Gate that routes each token between A and B

## Repository Structure
