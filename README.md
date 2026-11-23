# 🧪 LLM & SLM Testgen Benchmark — Replication Package

This repository contains the complete replication package—including source code, generated test suites, manual inspection logs, datasets, and inference scripts—for the study:

**"LLMs and SLMs as Test Generators: An In-Depth Exploration of Generated Tests"**  
*Submitted to IEEE Access (2026)*  
*(Extending our preliminary study: "LLMs as Test Generators: A Comparative Benchmarking Study", SBES 2025)*

> 📄 **Preliminary Paper (SBES 2025):** [esdrascaleb.github.io/llm-testgen-benchmark](https://esdrascaleb.github.io/llm-testgen-benchmark/)

---

## 🔧 Inference Hyperparameters & Execution Configuration

To guarantee full experimental replicability across local consumer hardware and HPC environments, all experiments in this benchmark use standardized OpenAI-compatible API calls (`main.py` / `request.py`) with the following decoding preset (aligned with the original ChatTester configuration):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **`temperature`** | `0.7` | Controls sampling randomness during candidate test generation and repair. |
| **`max_tokens`** | `1024` | Maximum number of new tokens generated per inference response (`max_new_tokens`). |
| **`max_prompt_tokens`** | `3000` | Maximum context window budget allocated for the focal class and method prompt. |
| **`batch_size`** | `1` | Sequential request execution (`batch_size = 1`), ensuring hardware-agnostic determinism between local CPUs and datacenter GPUs. |
| **Max Repair Attempts** | `5` | Maximum self-correction iterations per compilation failure (`Step 5` in ChatTesterMut). |
| **Global Retry Limit** | `5` | Maximum full pipeline restarts per target method (`Step 11` in ChatTesterMut). |

### 🖥️ Evaluated Hardware Environments
Because inference is decoupled via OpenAI-compatible local servers (vLLM / Ollama) using `batch_size = 1`, test generation logic behaves identically regardless of the underlying accelerator:
* **Local Consumer Baseline:** Intel Core i7 (11th Gen), 16 GB RAM, CPU-only inference (average footprint: `6.56 GB RAM`, `52% CPU`, `3.6 min/test`).
* **HPC Batch Acceleration (NPAD/UFRN):** Multi-model batch evaluations across all SF110 target methods were accelerated using:
  * **NVIDIA Tesla V100 Nodes:** 8× NVIDIA Tesla V100-SXM2 (**16 GB VRAM** per GPU), 2× Intel Xeon E5-2683 v4, 512 GB DDR4 RAM.
  * **NVIDIA H200 Nodes:** 4× NVIDIA H200 (**141 GB HBM3e VRAM** per GPU), 2× Intel Xeon Gold 6548N, 1 TB DDR5 RAM.

---

## 🧾 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `generated_tests/` | Automatically generated test suites from the 18 evaluated models (6 LLMs & 12 SLMs) and prompt sensitivity experiments. |
| `src/main/resources/dependency/` | Java dependencies required to compile and execute the SF110 target projects. |
| `chatunitest-standalone.jar` | Custom **ChatTesterMut** pipeline integrating OpenAI-compatible endpoints and the 6-operator AST mutation engine. |
| `TestSmellDetector.jar` | **TsDetect** tool used to identify structural test smells and compute the *Upper Bound Smelly Rate* ($R_{smell}^{ub}$). |
| `finalcompleddata.xlsx` | Complete raw and normalized datasets for RQ1 (discarded test diagnostics), RQ2 (18-model benchmark), RQ3 (blind human evaluation & Spearman correlations), and RQ4 (SLM prompt sensitivity). |
| `usages.csv` | Empirical local CPU/RAM usage and latency telemetry logs. |
| `main.py` | Starts the local vLLM / OpenAI-compatible inference server with integrated test generation parameters (`temperature=0.7`, `max_tokens=1024`). |
| `request.py` | Dispatches prompt requests (`max_prompt_tokens=3000`, `batch_size=1`) to ChatTesterMut and handles responses. |
| `experiment.py` | Interactive runner script to execute benchmarks across local and web-based models. |
| `benchmarkscript.sh` | Automated batch script for HPC / multi-model execution using generated `.env` files. |
| `generate_env_model.py` | Generates model-specific `.env` configuration files from `template.env`. |
| `generatecsvtotest.py` | Aggregates execution logs, test smells, and cyclomatic complexity into unified CSV reports. |
| `auxfunctions.py` | Auxiliary Python utilities for server orchestration and metric parsing. |
| `testmodel.py` | Sanity-check validation script (Java Fibonacci & SF110 smoke test). |
| `models.txt` / `models_local.txt` / `models_web.txt` | Lists of the 18 evaluated architectures (12 open-weight SLMs and 6 proprietary/web LLMs). |
| `template.env` | Environment variable template for inference endpoints and decoding parameters. |
| `requirements.txt` | Pinned Python dependencies and versions for exact environment reproduction. |
| `LICENSE` | Mozilla Public License Version 2.0. |

---

## ⚙️ System Requirements

* **Java:** OpenJDK 11 or later (with Maven/Ant support for SF110 targets)
* **Python:** Version **3.10** (strongly recommended; Python 3.13 is not supported by older PyTorch/vLLM builds)
* **Python Packages:** Pinned in `requirements.txt`
* **Memory:** At least **8 GB RAM** recommended for running 1B–3B local SLMs on CPU (16 GB RAM recommended for 3B models)
* **Disk Space:** ~30 GB for model weights, SF110 projects, and generated test artifacts

---

## 🚀 Installation & Environment Setup

### 1. Create and activate an isolated Conda environment
```bash
conda create -n localllm python=3.10 -y
conda activate localllm

```

### 2. Clone the repository

```bash
git clone [https://github.com/EsdrasCaleb/localLLM](https://github.com/EsdrasCaleb/localLLM) -b llm-testgen-benchmark
cd localLLM

```

### 3. Install pinned Python dependencies

```bash
pip install -r requirements.txt

```

---

## ▶️ Running the Benchmark

### Interactive Execution (Single Model / Local Validation)

To run the generation pipeline interactively:

```bash
python experiment.py

```

This script prompts for model selection, initializes the inference client with `temperature=0.7` and `max_tokens=1024`, executes ChatTesterMut, and logs all passing and discarded tests.

### Batch / HPC Execution (Multi-Model Benchmark)

To reproduce the full multi-model benchmark in a batch or SLURM/HPC environment:

1. Generate `.env` files for all target models defined in `models.txt`:
```bash
python generate_env_model.py

```


2. Launch the automated batch pipeline:
```bash
bash benchmarkscript.sh

```



---

## 📜 License & Related Repositories

This artifact is distributed under the terms of the **Mozilla Public License Version 2.0**.

* **Main Repository Branch:** [localLLM/tree/llm-testgen-benchmark](https://github.com/EsdrasCaleb/localLLM/tree/llm-testgen-benchmark)
* **SBES 2025 Release Tag:** [localLLM/releases/tag/sbes2025](https://github.com/EsdrasCaleb/localLLM/releases/tag/sbes2025)
* **ChatTesterMut Core Library:** [`chatunitest-core`](https://github.com/EsdrasCaleb/chatunitest-core)
* **Standalone CLI Tool:** [`chatunitest-standalone`](https://github.com/EsdrasCaleb/chatunitest-standalone)
