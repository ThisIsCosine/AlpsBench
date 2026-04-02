# AlpsBench

> [Accepted by SIGIR 2026 Resource Track] [April 2026]

<p align="center">
  <img src="figures/github_img.png" alt="AlpsBench overview" width="960" />
</p>

<p align="center">
  <strong>AlpsBench</strong> is a benchmark for long-term personalization in LLM assistants on real dialogue data.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.26680"><img src="https://img.shields.io/badge/Paper-arXiv%3A2603.26680-B31B1B" alt="Paper" /></a>
  <a href="https://huggingface.co/datasets/Cosineyx/Alpsbench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFB000" alt="Hugging Face Dataset" /></a>
  <a href="docs/leaderboard.md"><img src="https://img.shields.io/badge/Leaderboard-Policy-green" alt="Leaderboard" /></a>
  <a href="docs/usage.md"><img src="https://img.shields.io/badge/Docs-Usage-black" alt="Docs" /></a>
  <a href="adapter_example/"><img src="https://img.shields.io/badge/Adapter-Example-orange" alt="Adapter Example" /></a>
</p>

## ✨ Overview

AlpsBench evaluates long-term personalization across four core tasks:

- **Task 1: memory extraction**
- **Task 2: memory update**
- **Task 3: memory retrieval**
- **Task 4: memory-grounded response generation**

It is built on real-world human-LLM dialogues curated from WildChat and pairs those dialogues with human-verified structured memories. The public release is designed so users can either:

- score their own `predictions.jsonl`, or
- plug their own model into AlpsBench through `--predict-program` with a simple `stdin/stdout` JSON adapter

If you are new to AlpsBench, the fastest reading order is:

1. [docs/usage.md](docs/usage.md)
2. [docs/prediction_contract.md](docs/prediction_contract.md)
3. [docs/evaluation_metric.md](docs/evaluation_metric.md)

## 🧭 Benchmark Design

<p align="center">
  <img src="figures/tasks_outline.jpg" alt="Evaluation Tasks Overview" />
</p>

At a high level, AlpsBench measures the full memory lifecycle:

- **Extraction:** can the model turn dialogue into structured memories?
- **Update:** can it revise memories when user preferences change?
- **Retrieval:** can it pick the right memory from large distractor pools?
- **Utilization:** can it generate responses that are actually grounded in remembered user information?

## 📊 Benchmark Results

The paper reports the following reference results for representative general-purpose models.

These numbers are benchmark-side paper results, not the direct output of the public `scripts/evaluate.py` local scorer.


| Model                 | Task 1<br>Extraction | Task 2<br>Update | Task 3 Retr.<br>100 | Task 3 Retr.<br>300 | Task 3 Retr.<br>500 | Task 3 Retr.<br>700 | Task 3 Retr.<br>1000 | Task 4<br>PA | Task 4 PF<br>Gen. | Task 4 PF<br>Int. | Task 4<br>VRA | Task 4<br>CF | Task 4 EI<br>EN | Task 4 EI<br>CN |
| :-------------------- | :------------------: | :--------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :------------------: | :----------: | :---------------: | :---------------: | :-----------: | :----------: | :-------------: | :-------------: |
| **GPT-5.2**           |        41.43        |    **81.49**    |       0.9254       |       0.9052       |       0.8884       |       0.8733       |        0.8572        |    0.5702    |      0.6983      |    **0.7680**    |    0.5702    |    0.5702    |      3.42      |      3.90      |
| **GPT-4.1-mini**      |        33.69        |      54.66      |       0.8802       |       0.8156       |       0.7761       |       0.7482       |        0.7295        |    0.4018    |      0.4995      |      0.5240      |    0.4018    |    0.4018    |      2.79      |      2.92      |
| **DeepSeek Reasoner** |        47.79        |      80.91      |     **0.9569**     |     **0.9484**     |     **0.9376**     |       0.9083       |      **0.9273**      |    0.5825    |      0.6483      |      0.6120      |    0.5825    |  **0.9602**  |      3.66      |    **4.00**    |
| **Gemini-3 Flash**    |      **51.67**      |      68.85      |       0.9538       |       0.9419       |       0.9342       |     **0.9268**     |        0.9269        |  **0.6895**  |    **0.7655**    |      0.7052      |  **0.6895**  |    0.8328    |      3.49      |      3.58      |
| **Llama-4 Maverick**  |        22.07        |      58.84      |       0.8729       |       0.6005       |       0.5616       |       0.5141       |        0.4811        |    0.2684    |      0.1152      |      0.3080      |    0.1552    |    0.8720    |      2.48      |      2.38      |
| **Claude-Sonnet-4.5** |        41.64        |      51.25      |       0.9542       |       0.9030       |       0.9222       |       0.8999       |        0.8855        |    0.5614    |      0.6045      |      0.5498      |    0.5933    |    0.9514    |      3.10      |      3.05      |
| **Qwen3-max**         |        39.01        |      76.28      |       0.9180       |       0.8669       |       0.8314       |       0.7871       |        0.7542        |    0.6228    |      0.6901      |      0.6574      |    0.6834    |    0.8267    |    **3.68**    |      3.84      |

Notes:

- **PA** = Persona Awareness
- **PF** = Preference Following
- **VRA** = Virtual-Reality Awareness
- **CF** = Constraint Following
- **EI** = Emotional Intelligence

For reporting policy and submission details, see [docs/leaderboard.md](docs/leaderboard.md).

## 🚀 Quickstart

Create an environment and install dependencies:

```bash
conda create -n alpsbench python=3.10
conda activate alpsbench
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Validate the released data layout and smoke-check the example split:

```bash
python scripts/validate_data.py
python scripts/smoke_examples.py
python scripts/evaluate.py --task task1 --split examples --oracle
```

Run your own predictions on public local-scoring splits:

```bash
python scripts/evaluate.py --task task1 --split dev --predictions my_task1_predictions.jsonl
python scripts/evaluate.py --task task1 --split validation --predictions my_task1_predictions.jsonl
python scripts/evaluate.py --task task3 --distractors 500 --split dev --predictions my_task3_predictions.jsonl
python scripts/evaluate.py --task task4 --ability ability3 --split dev --predictions my_task4_predictions.jsonl
```

Generate official `test` predictions for submission:

```bash
python scripts/evaluate.py --task task1 --split test --predictions my_task1_predictions.jsonl
python scripts/evaluate.py --task task4 --ability ability2 --split test --predict-program python --predict-arg my_adapter.py
```

For `test`, the evaluator writes `predictions.jsonl` and `summary.json` but does not score locally because no public references are available.

For `examples`, `dev`, and `validation`, `scripts/evaluate.py` returns public local proxy scores for self-checking and debugging. These local scores are useful for format validation and iteration, but they are not identical to the full benchmark-side reporting used in the paper and leaderboard. See [docs/metrics.md](docs/metrics.md) and [docs/evaluation_metric.md](docs/evaluation_metric.md) for the task-by-task scoring surface.

**Evaluation note.** To preserve blind evaluation integrity, we do not fully release benchmark-side judge prompts or hidden scoring details. The public task logic, prediction contract, and local evaluation surface are documented in [docs/evaluation_metric.md](docs/evaluation_metric.md) and related docs.

## 🔌 Adapter Workflow

The public CLI is provider-agnostic. If you want AlpsBench to call your model directly, use `--predict-program` with repeated `--predict-arg`.

Example:

```bash
python scripts/evaluate.py --task task4 --ability ability2 --split examples --predict-program python --predict-arg adapter_example/minimal_adapter.py
```

The evaluator sends one `model_input.jsonl` row as JSON on `stdin`, and your adapter must return exactly one prediction JSON object on `stdout`.

The repository includes a minimal standalone skeleton in [adapter_example/](adapter_example/):

- [adapter_example/README.md](adapter_example/README.md)
- [adapter_example/minimal_adapter.py](adapter_example/minimal_adapter.py)

On Windows, if your adapter emits non-ASCII JSON directly to `stdout`, prefer:

```bash
python scripts/evaluate.py --task task4 --ability ability2 --split examples --predict-program python --predict-arg -X --predict-arg utf8 --predict-arg adapter_example/minimal_adapter.py
```

Python's default `json.dump(...)` is already safe because it emits ASCII escapes unless you explicitly disable that behavior.

## 📦 Public Release Layout

The public repository surface is:

- `benchmark_data/`
- `adapter_example/`
- `scripts/`
- `src/benchmark/`
- `docs/`

Private maintainer assets such as hidden gold, local tools, and notebooks are not part of the benchmark interface.

Public benchmark data is organized by split:

- `benchmark_data/examples/`: tiny runnable examples with public references
- `benchmark_data/dev/`: public development split with public references
- `benchmark_data/validation/`: public holdout validation split with public references
- `benchmark_data/test/`: official public test inputs only
- `benchmark_data/artifacts/`: release manifests and split summaries

Each public track uses:

- `model_input.jsonl`: model-visible input rows
- `reference_output.jsonl`: public gold rows for `examples`, `dev`, and `validation`

`benchmark_data/test/` does not expose public `reference_output.jsonl`. Private hidden gold for `test` lives under `hidden/private_gold/`.

The released split policy is deterministic and track-local:

- `1/5` of each track is in `dev`
- `1/5` of each track is in `validation`
- `3/5` of each track is in `test`

Current row counts are recorded in `benchmark_data/artifacts/build_summary.json`, and split metadata is recorded in `benchmark_data/artifacts/split_manifest.json`.

## 🧾 Prediction And Submission Contract

Users submit prediction rows, not `reference_output.jsonl`.

- prediction files must contain exactly one JSON object per input row
- every input `benchmark_id` must appear exactly once
- `benchmark_data/examples/*/reference_output.jsonl` is public gold and sample data, not the required submission format
- `runs/public/test/<track>/predictions.jsonl` is the run artifact to send for hidden evaluation

The exact task-level prediction schema is documented in [docs/prediction_contract.md](docs/prediction_contract.md). That document also spells out the nested object structure for `memory_items` and the expected content shape for Task 3 and Task 4 outputs.

## 🏆 Leaderboard And Submission

The public reporting surface follows the paper-style task view:

- `task1`: memory extraction
- `task2`: memory update
- `task3`: retrieval
- `task4_ability1`
- `task4_ability2`
- `task4_ability3`
- `task4_ability4`
- `task4_ability5`

Task 4 abilities are scored and ranked independently

The paper reports reference Task 3 retrieval baselines including `nltk + bm25` and `all-MiniLM-L6-v2`. The current public repository release does not bundle official runnable baseline implementations; baseline details and reported results should be read from the paper.

For external submission:

1. run your system on `benchmark_data/test/`
2. produce a contract-conforming `predictions.jsonl`
3. submit the generated output by contacting the first author listed in the paper or by opening a GitHub issue

After verification, accepted model results can be added to the benchmark reporting surface.

## 🗂️ Public Commands

User-facing commands:

- `python scripts/validate_data.py`
- `python scripts/smoke_examples.py`
- `python scripts/evaluate.py --task ... --split ... --dry-run`
- `python scripts/evaluate.py --task ... --split ... --oracle`
- `python scripts/evaluate.py --task ... --split ... --predictions ...`
- `python scripts/evaluate.py --task task3 --distractors 100|300|500|700|1000 --split ...`
- `python scripts/evaluate.py --task ... --split ... --predict-program ... --predict-arg ...`
- `python scripts/evaluate.py --task ... --split ... --predict-command "..."`

Maintainer-oriented commands:

- `python scripts/build_data.py --overwrite`
- `python scripts/split_public_data.py --force`

## 📚 Documentation Map

- [docs/usage.md](docs/usage.md)
- [docs/prediction_contract.md](docs/prediction_contract.md)
- [docs/data.md](docs/data.md)
- [docs/metrics.md](docs/metrics.md)
- [docs/evaluation_metric.md](docs/evaluation_metric.md)
- [docs/leaderboard.md](docs/leaderboard.md)
- [docs/repository_structure.md](docs/repository_structure.md)
- [docs/implementation_blueprint.md](docs/implementation_blueprint.md)
- [REPOSITORY_REFACTOR_ROADMAP.md](REPOSITORY_REFACTOR_ROADMAP.md)

## 🧪 Dataset Card And Provenance

At a high level, the released benchmark data follows the paper's benchmark construction pipeline and should be read together with the paper for full details.

- source provenance: the benchmark is derived from real-world human-LLM dialogues curated from WildChat
- scale: the paper describes AlpsBench as comprising 2,500 long-term interaction sequences
- memory quality: the benchmark pairs those interactions with human-verified structured memories
- annotation process: the paper describes a four-step pipeline of data collection, structured memory extraction, human annotation or verification, and task construction
- release form: this repository distributes the processed benchmark package used for public evaluation rather than raw dialogue dumps

<p align="center">
  <img src="figures/data_construction.png" alt="Data Construction Pipeline" />
</p>

`benchmark_data/` in this repository is the released public benchmark dataset for the current `v1` release. There is no separate public download for a larger benchmark package in the current repository release.

This means:

- `benchmark_data/` is the final public benchmark data layout
- `benchmark_data/artifacts/` contains committed release metadata for that data
- `runs/public/...` contains user-generated evaluation artifacts, not dataset files

## 🔐 Data Usage And Privacy

Please treat `benchmark_data/` as benchmark data for research and evaluation use, not as a source for deanonymization, identity linkage, or recovery of hidden evaluation targets.

For licensing, attribution, provenance, and additional methodological details, follow the paper and the upstream WildChat source cited there.

Repository-level files:

- [LICENSE](LICENSE)
- [DATA_USAGE.md](DATA_USAGE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CHANGELOG.md](CHANGELOG.md)

## ⚙️ Configuration

API-backed evaluation should start from `configs/api.example.json`.

## 📖 Citation

```bibtex
@article{xiao2026alpsbench,
  title={AlpsBench: An LLM Personalization Benchmark for Real-Dialogue Memorization and Preference Alignment},
  author={Xiao, Jianfei and Yu, Xiang and Wang, Chengbing and Zheng, Wuqiang and Lin, Xinyu and Liu, Kaining and Ding, Hongxun and Zhang, Yang and Wang, Wenjie and Feng, Fuli and He, Xiangnan},
  journal={arXiv preprint},
  year={2026}
}
```
