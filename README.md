# 🧪 LLM Testgen Benchmark — Artifact for SBES 2025

This repository contains the code, data, scripts, and experimental setup used in the paper:

**"LLMs as Test Generators: A Comparative Benchmarking Study"**
*Accepted at SBES 2025 (Research Track)*
> [📄 Link to the paper (PDF)](https://esdrascaleb.github.io/llm-testgen-benchmark/)

---

## 🧾 Repository Structure

| File/Folder                      | Description                                                             |
| -------------------------------- | ----------------------------------------------------------------------- |
| `generated_tests/`               | Automatically generated tests from the paper benchmark.                 |
| `src/main/resources/dependency/` | Java dependencies used by the system.                                   |
| `.gitignore`                     | Git ignore file.                                                        |
| `LICENSE`                        | Mozilla Public License Version 2.0                                 |
| `README.md`                      | This file.                                                              |
| `TestSmellDetector.jar`          | TSDetect tool used to identify test smells.                             |
| `chatunitest-standalone.jar`     | Main ChatTesterMut tool used to simulate LLM-based test generation.     |
| `auxfunctions.py`                | Auxiliary Python functions used by the local LLM server.                |
| `benchmarkscript.sh`             | Script to run all benchmarks using the generated environments.          |
| `generate_env_model.py`          | Script to generate `.env` configuration files used during benchmarking. |
| `experiment.py`                  | Main script to run experiments (calls LLMs and prompts user).           |
| `finalcompleddata.xlsx`          | All experimental data used and reported in the paper.                   |
| `generatecsvtotest.py`           | Generates a CSV combining tests, smells, and complexity data.           |
| `main.py`                        | Starts the local VLLM server with integrated test generation.           |
| `models.txt`                     | List of all models (local and remote) used in `generate_env_model.py`.  |
| `models_local.txt`               | List of LLMs installed locally for benchmarking.                        |
| `models_web.txt`                 | List of web-based endpoints used in benchmarking.                       |
| `request.py`                     | Sends a prompt request to ChatTesterMut and handles response.           |
| `requirements.txt`               | Python dependencies (with versions).                                    |
| `template.env`                   | Template for `.env` file used in `generate_env_model.py`.               |
| `testmodel.py`                   | Quick test script to evaluate a model's ability to generate tests.      |
| `usages.csv`                     | Local usage metrics generated from experiments (can be reproduced).     |

---

## ⚙️ Requirements

* **Java**: OpenJDK 11 or later
* **Python**: Version **3.10** (strongly recommended, Python 3.13 is **not supported by PyTorch**)
* **pip packages**: All listed in `requirements.txt`
* **Memory**: At least 8 GB RAM recommended to run local LLMs
* **Disk Space**: \~30GB for downloaded models and generated data
* **Recommended**: [Anaconda](https://www.anaconda.com/products/distribution) (especially on Windows)

---

## 🚀 Installation (Recommended: Anaconda on Windows/Linux/macOS)

### 1. Download and install Anaconda

* Go to [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
* Download the version for **Python 3.10** (or the default one, usually 3.10)
* Install it using default settings (check “Add Anaconda to PATH” if prompted)

### 2. Create and activate an isolated environment

Open the **Anaconda Prompt** (or terminal in Linux/macOS) and run:

```bash
conda create -n localllm python=3.10
conda activate localllm
```

### 3. Clone the repository and navigate into it

```bash
git clone https://github.com/EsdrasCaleb/localLLM -b llm-testgen-benchmark
cd localLLM
```

### 4. Install the Python dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Experiment

To run the full benchmark process:

```bash
python experiment.py
```

This script will:

* Prompt for any required model selection
* Handle both local and web-based models
* Collect and save the results

---

## ⚙️ Optional: Advanced usage

If you plan to execute the experiments in a batch-processing or HPC environment:

### Generate the environment configuration manually

```bash
python generate_env_model.py
```

### Then execute the benchmark script directly

```bash
bash benchmarkscript.sh
```

---

## ✅ Completeness

* The file `finalcompleddata.xlsx` contains all results reported in the paper.
* The repository includes all scripts, models, and automation to fully reproduce the benchmark.
* Meets requirements for the **Available** and **Functional** badges.

---

## 📜 License

This artifact is distributed under the terms of the **Mozilla Public License Version 2.0**.

---

## 🔗 Also Available on GitHub

* Main repository branch: [localLLM/tree/llm-testgen-benchmark](https://github.com/EsdrasCaleb/localLLM/tree/llm-testgen-benchmark)
* Release for SBES 2025: [localLLM/releases/tag/sbes2025](https://github.com/EsdrasCaleb/localLLM/releases/tag/sbes2025)
* ChatTesterMut core library: [`chatunitest-core`](https://github.com/EsdrasCaleb/chatunitest-core)
* Standalone CLI tool: [`chatunitest-standalone`](https://github.com/EsdrasCaleb/chatunitest-standalone)

