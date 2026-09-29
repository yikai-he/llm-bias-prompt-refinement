# LLM Bias Prompt Refinement

An LLM-based pipeline for evaluating and improving gender-bias detection prompts. It evaluates target models on implicit-bias samples, measures accuracy, balanced accuracy, and MCC, and uses a second LLM to refine prompts for misclassified examples.

## Project Materials

- [Project report](docs/project-report.pdf)
- [Project poster](docs/project-poster.pdf)

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp agent_pipeline/.env.example agent_pipeline/.env
```

Add the API keys required by your providers to `agent_pipeline/.env`, then set the providers, model names, and `REASON` option in `agent_pipeline/config/settings.py`.

## Usage

Run scripts from their directory so relative paths resolve correctly:

```bash
cd agent_pipeline/scripts
python run_baseline.py
```

To refine failed prompts, set `BASELINE_TIMESTAMP` and `LOOP` in `run_refinement_loop.py`, then run:

```bash
python run_refinement_loop.py
```

Datasets are stored in `dataset/`, generated results in `test_results/`, and selected project documents in `docs/`. Historical experiment scripts are retained under `archive/`.
