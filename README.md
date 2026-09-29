# LLM Bias Prompt Refinement

An LLM-based pipeline for evaluating and improving gender-bias detection prompts. It runs a target model on implicit-bias samples, measures accuracy, balanced accuracy, and MCC, then uses a second LLM to refine prompts for misclassified examples.

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r agent_pipeline/requirements.txt
pip install langchain-openai pandas matplotlib
```

Create `agent_pipeline/.env` with the keys required by your selected providers:

```env
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
NOVITA_API_KEY=...
NOVITA_API_URL=https://api.novita.ai/v3/openai
```

Set the model providers, model names, and `REASON` option in `agent_pipeline/config/settings.py`.

## Usage

Run scripts from their directory so relative data and output paths resolve correctly:

```bash
cd agent_pipeline/scripts
python run_baseline.py
```

To refine failed baseline prompts, set `BASELINE_TIMESTAMP` and `LOOP` in `run_refinement_loop.py`, then run:

```bash
python run_refinement_loop.py
```

Results are written to `test_results/agent_results/`. Datasets are stored in `dataset/`, and plots can be generated with `python run_plot.py`.
