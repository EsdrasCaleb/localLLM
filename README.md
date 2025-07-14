# LLM Testgen Benchmark — Artifact for SBES 2025

This repository contains the **code, data, scripts, and experimental setup** used in the paper:

> **"Automatic Test Generation and Smell Detection Using LLMs: Evaluating the ChatTesterMut Approach"**  
> Accepted at SBES 2025 (Research Track)  
> [📄 Link to the paper (PDF)](https://esdrascaleb.github.io/llm-testgen-benchmark/)

---

## 🧾 Repository Structure

| File/Folder                     | Description |
|--------------------------------|-------------|
| `generated_tests/`             | Automatically generated tests from the paper benchmark. |
| `src/main/resources/dependency/` | Java dependencies used by the system. |
| `.gitignore`                   | Git ignore file. |
| `LICENSE`                      | Open-source license for this artifact. |
| `README.md`                    | This file. |
| `TestSmellDetector.jar`        | TSDetect tool used to identify test smells. |
| `chatunitest-standalone.jar`   | Main ChatTesterMut tool used to simulate LLM-based test generation. |
| `auxfunctions.py`              | Auxiliary Python functions used by the local LLM server. |
| `benchmarkscript.sh`           | Script to run all benchmarks using the generated environments. |
| `generate_env_model.py`        | Script to generate `.env` configuration files used during benchmarking. |
| `experiment.py`                | Main script to run experiments (calls LLMs and prompts user when needed). |
| `finalcompleddata.xlsx`        | **All experimental data** used and reported in the paper. |
| `generatecsvtotest.py`         | Generates a CSV combining tests, smells, and complexity data. |
| `main.py`                      | Starts the local VLLM server with integrated test generation endpoints. |
| `models.txt`                   | List of all models (local and remote) used in `generate_env_model.py`. |
| `models_local.txt`             | List of LLMs installed locally for benchmarking. |
| `models_web.txt`               | List of web-based endpoints used in benchmarking. |
| `request.py`                   | Sends a prompt request to ChatTesterMut and handles response. |
| `requirements.txt`             | Python dependencies (with versions). |
| `template.env`                 | Template for `.env` file used in `generate_env_model.py`. |
| `testmodel.py`                 | Quick test script to evaluate a model's ability to generate tests. |
| `usages.csv`                   | Local usage metrics generated from experiments (can be reproduced). |

---

## ⚙️ Requirements

- **Java**: OpenJDK 8 or later
- **Python**: Version 3.8+
- **pip packages**: All listed in `requirements.txt`
- **Memory**: At least 8 GB RAM recommended to run local LLMs
- **Disk Space**: ~5GB for downloaded models and generated data
- Optional: Docker (to isolate Python environment)

---

## 🚀 Installation

1. Unpack the archive downloaded from Zenodo.

2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main experiment script:

   ```bash
   python experiment.py
   ```

   This script will:

   * Automatically prompt for any required model selection.
   * Handle local or web-based models (if needed).
   * Collect all results and save them accordingly.

4. (Optional) If you plan to execute the experiments in a batch-processing environment, you can:

   * Generate the environment files manually:

     ```bash
     python generate_env_model.py
     ```

   * Then execute the benchmark script directly:

     ```bash
     bash benchmarkscript.sh
     ```

These optional scripts give you more control over the execution in environments like HPC clusters, but for most users, simply running `experiment.py` is enough.

---

## ✅ Completeness

* The `finalcompleddata.xlsx` file contains **all data reported** in the accepted article.
* The repository includes **scripts, models, and automation** to fully reproduce the benchmark.
* We aim to meet the requirements for both the **Available** and **Functional** badges.

---
## 📜 License

This artifact is distributed under the terms of the [Mozilla Public License Version 2.0](LICENSE).
