## Lien vers nos découvertes
[Sous forme d'un article de 10 pages](./Paper.pdf?raw=true)

[English version of the paper](./english_paper.pdf?raw=true)

[Le rapport complet de 42 pages](./rapport.pdf?raw=true)

## Mémoire paramétrique vs mémoire factuelle des LLM (projet DS-1000 + Numpy)

This repository implements a full research pipeline to measure **whether LLMs rely on their parametric memory** (knowledge baked in during pre-training) **or on documentation provided in context** (factual / RAG memory) when generating Python code.

---

## Table of Contents

1. [Research Concept](#1-research-concept)
2. [Repository Layout](#2-repository-layout)
3. [End-to-End Pipeline](#3-end-to-end-pipeline)
4. [Source Code — `src/`](#4-source-code--src)
   - [Orchestrator — `execution_process.py`](#41-orchestrator--execution_processpy)
   - [LLM Client — `llmclient.py`](#42-llm-client--llmclientpy)
   - [Code Cleaner — `cleaning.py`](#43-code-cleaner--cleaningpy)
   - [Counterfactual Wrappers — `Wrap*.py`](#44-counterfactual-wrappers--wrappy)
   - [AST Normalizers — `ast_cleaning_*.py`](#45-ast-normalizers--ast_cleaning_py)
5. [Documentation Generation — `src/documentation/`](#5-documentation-generation--srcdocumentation)
   - [Real NumPy Documentation](#51-real-numpy-documentation)
   - [Counterfactual Documentation Generators](#52-counterfactual-documentation-generators)
   - [Noisy Documentation Generator](#53-noisy-documentation-generator)
   - [Generated Documentation Files](#54-generated-documentation-files)
6. [Dataset & Data Files — `data/`](#6-dataset--data-files--data)
   - [Dataset Format (DS-1000 JSONL)](#61-dataset-format-ds-1000-jsonl)
   - [Dataset Corruption Scripts](#62-dataset-corruption-scripts)
7. [Configuration Files — `configs/`](#7-configuration-files--configs)
   - [Full Configuration Schema](#71-full-configuration-schema)
   - [Available Configs](#72-available-configs)
8. [Running the Pipeline](#8-running-the-pipeline)
   - [Prerequisites](#81-prerequisites)
   - [Basic Execution](#82-basic-execution)
   - [Filtering Options](#83-filtering-options)
   - [SLURM / HPC Execution](#84-slurm--hpc-execution)
9. [Results Format](#9-results-format)
10. [Analysis & Visualization — `src/perf_review/`](#10-analysis--visualization--srcperf_review)
    - [`plot_model_perf.py`](#101-plot_model_perfpy)
    - [`plot_noisy_doc_perf.py`](#102-plot_noisy_doc_perfpy)
    - [`sanity_check.py`](#103-sanity_checkpy)
    - [`advanced_sanity_check.py`](#104-advanced_sanity_checkpy)
11. [Interpreting the Metrics](#11-interpreting-the-metrics)
12. [Creating a New Experiment](#12-creating-a-new-experiment)

---

## 1. Research Concept

The core idea is to present an LLM with a **counterfactual version of NumPy** — a library that behaves identically to NumPy but whose function names have been systematically renamed. Three renaming schemes are tested:

| Scheme | Example | Wrapper module |
|--------|---------|----------------|
| `_v2` suffix | `np.mean_v2()`, `np.array_v2()` | `WrapV2Numpy` |
| Trailing `_` | `np.mean_()`, `np.array_()` | `WrapUnderscoreNumpy` |
| Capitalization | `np.Mean()`, `np.Array()` | `WrapCapitalizeNumpy` |

Matching counterfactual documentation is injected into the LLM's context window. The model is then evaluated on **DS-1000** coding problems (NumPy subset, 159 tasks):

- **Injection mode**: The LLM receives the fake documentation and the corrupted problem statement. Its generated code is executed **twice**:
  - With the counterfactual wrapper library → metric: `passed`
  - With real NumPy → metric: `control_passed`
- **Control mode**: The LLM receives real NumPy documentation and solves unmodified problems.

The key interpretive insight:

| `passed` | `control_passed` | Interpretation |
|----------|-----------------|----------------|
| ✅ | ❌ | LLM genuinely followed the injected documentation |
| ✅ | ✅ | LLM bypassed the perturbation (suspicious — task may not need modified API) |
| ❌ | ✅ | LLM fell back on parametric memory (used real API, failed with wrapper) |
| ❌ | ❌ | Task too hard or code quality issue |

---

## 2. Repository Layout

```
memory_code_eval/
├── src/                            # All pipeline source code
│   ├── execution_process.py        # Main orchestrator (entry point)
│   ├── llmclient.py                # Ollama LLM client
│   ├── cleaning.py                 # LLM output cleaning & normalization
│   ├── WrapV2Numpy.py              # Counterfactual wrapper: _v2 suffix
│   ├── WrapUnderscoreNumpy.py      # Counterfactual wrapper: trailing _
│   ├── WrapCapitalizeNumpy.py      # Counterfactual wrapper: capitalization
│   ├── ast_cleaning_v2.py          # AST normalizer for _v2
│   ├── ast_cleaning_underscore.py  # AST normalizer for _
│   ├── ast_cleaning_capitalize.py  # AST normalizer for capitalize
│   ├── capitalize_numpy_documentation.py  # (standalone) Capitalize doc generator
│   ├── underscore_numpy_documentation.py  # (standalone) Underscore doc generator
│   ├── documentation/              # All documentation files & generators
│   │   ├── numpy_documentation_creation.py     # Real NumPy doc generator
│   │   ├── v2_numpy_documentation.py            # _v2 doc generator
│   │   ├── underscore_numpy_documentation.py    # _ doc generator
│   │   ├── capitalize_numpy_documentation.py    # Capitalize doc generator
│   │   ├── noisy_doc_generator.py               # Noisy documentation generator
│   │   ├── documentation_dictionnary.py         # Shared doc discovery utilities
│   │   ├── real_minimal_numpy.txt               # Real NumPy (minimal)
│   │   ├── real_ultra_minimal_numpy.txt         # Real NumPy (ultra-minimal)
│   │   ├── real_full_doc_numpy.txt              # Real NumPy (full)
│   │   ├── corrupted_minimal_numpy_v2.txt       # Fake _v2 (minimal, no noise)
│   │   ├── corrupted_minimal_numpy_v2_noise25.txt  # Fake _v2 (minimal, 25% noise)
│   │   ├── ... (all noise/variant combos)
│   │   ├── interest_functions_v2.txt            # Functions actually used in DS-1000 (_v2)
│   │   ├── interest_functions_underscore.txt
│   │   ├── interest_functions_capitalize.txt
│   │   ├── doc_explanation.txt                  # Textual explanation of the modification
│   │   ├── doc_explanation_v2.txt
│   │   └── doc_explanation_capitalize.txt
│   └── perf_review/                # Analysis & plotting scripts
│       ├── plot_model_perf.py       # Global control vs injection comparison plots
│       ├── plot_noisy_doc_perf.py   # Noisy-doc performance histograms
│       ├── sanity_check.py          # Quality checks & anomaly detection
│       └── advanced_sanity_check.py # Additional diagnostics
├── data/                           # Datasets and data preparation scripts
│   ├── ds1000_npyOnly.jsonl                     # Original DS-1000 (NumPy, 159 tasks)
│   ├── ds1000_npyOnly_corrupted_v2.jsonl        # Corrupted (_v2 scheme)
│   ├── ds1000_npyOnly_corrupted_underscore.jsonl
│   ├── ds1000_npyOnly_corrupted_capitalize.jsonl
│   ├── data_modif_v2.py                         # Applies _v2 renaming to dataset
│   ├── data_modif_underscore.py
│   ├── data_modif_capitalize.py
│   ├── data_extraction.py                       # Extracts NumPy-only subset from full DS-1000
│   ├── get_none_numpy_tasks.py
│   ├── ds1000.jsonl                             # Full DS-1000 (all libraries)
│   └── data_smallsample.jsonl                   # Small sample for quick testing
├── configs/                        # Pre-built JSON configuration files
│   ├── config_bigexec_v2.json
│   ├── config_bigexec_underscore.json
│   ├── config_bigexec_capitalize.json
│   ├── config_v2_explanation.json
│   ├── config_underscore_explanation.json
│   └── config_capitalize_explanation.json
├── results/                        # All run outputs (created at runtime)
│   └── run_YYYY-MM-DD_HH-MM-SS/
│       ├── config.json             # Exact config snapshot
│       ├── results.jsonl           # One line per task
│       └── *.png                   # Generated plots
├── logs_slurm/                     # SLURM job logs
├── trick.sbatch                    # SLURM job script
├── requirements.txt
└── OLLAMA_MODELS_REFERENCE.md      # Reference of model names for Ollama
```

---

## 3. End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SETUP PHASE                                  │
│  Load JSON config → Create results/run_YYYY-MM-DD_HH-MM-SS/         │
│  Save config snapshot → Warm up Ollama + load models                │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
          For each (model, doc_name) pair in config:
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────────────────────────┐
│  CONTROL MODE   │               │         INJECTION MODE              │
│                 │               │                                     │
│ Dataset: origin │               │ Dataset: corrupted                  │
│ Doc: real NumPy │               │ Doc: counterfactual (+ maybe noisy) │
│ Lib: numpy      │               │ Lib: WrapXxxNumpy                   │
│                 │               │                                     │
│ LLM → code      │               │ For attempt 1..pass_at:             │
│ Clean code       │               │   LLM → raw response               │
│ Execute w/ numpy │               │   extract_code_and_fix()           │
│ passed = result │               │   normalize_object_attributes()     │
│ is_control=True │               │     ├─ Execute w/ fake lib          │
└────────┬────────┘               │     │  → passed                     │
         │                        │     └─ Execute w/ real numpy        │
         │                        │        → control_passed             │
         │                        │   If passed=True → stop retrying   │
         │                        └──────────────────┬──────────────────┘
         │                                           │
         └───────────────────┬───────────────────────┘
                             │
                             ▼
                    Write line to results.jsonl
                             │
                             ▼
               ┌─────────────────────────────┐
               │         ANALYSIS            │
               │  plot_model_perf.py         │
               │  plot_noisy_doc_perf.py     │
               │  sanity_check.py            │
               └─────────────────────────────┘
```

---

## 4. Source Code — `src/`

### 4.1 Orchestrator — `execution_process.py`

**Entry point** for every experiment. Run it from the project root:

```bash
python3 src/execution_process.py <config_file.json> [options]
```

Key responsibilities:

| Function | Description |
|----------|-------------|
| `load_config_from_path()` | Parses the JSON config |
| `setup_run_directory()` | Creates `results/run_.../`, saves config snapshot |
| `run_control()` | Iterates over `origin_data`, queries LLM with real doc, executes with real NumPy |
| `run_benchmark()` | Iterates over `corrupted_data`, queries LLM with counterfactual doc, double-executes (fake lib + real NumPy) |
| `execute_task_engine()` | Low-level subprocess execution with timeout, captures stdout/stderr, detects `SUCCESS_MARKER` |

**pass@k support**: set `exec.pass_at > 1` to allow multiple LLM attempts per task. The seed is incremented by 1 for each attempt. The first successful attempt's result is kept.

### 4.2 LLM Client — `llmclient.py`

Handles all communication with the **Ollama** local inference server.

```
LLMClient(config, model_name, doc_name, mode)
    mode = "control"   → uses real_lib system_prompt + real doc
    mode = "injection" → uses new_lib_injection system_prompt + counterfactual doc
```

Key behaviour:

- **Warm-up**: Polls `http://localhost:11434` until healthy, then fires a minimal query (`"Say OK."`) to ensure the model is loaded into GPU memory before the benchmark starts.
- **Documentation loading**: Reads the `path` file for the given `doc_name` and prepends the configured `intro` string. If `path` is empty (doc `"nothing"`), no documentation is injected.
- **Token counting**: Uses `prompt_eval_count` from Ollama's response; falls back to `len(prompt)/3` if not available.
- **Temperature & seed resolution**: Both can be a scalar (global) or a dict keyed by model name with a `"default"` fallback.

### 4.3 Code Cleaner — `cleaning.py`

Transforms the raw LLM text response into executable Python before it reaches the test harness.

**`extract_code_and_fix(raw_response)`** — main entry point:

```
Raw LLM text
    │
    ├─ Extract code from markdown fences (``` or <code> blocks)
    │
    ├─ FAST PATH (AST-based):
    │   ├─ Remove import statements
    │   ├─ Convert bare `return x` → `result = x`
    │   └─ Drop static constants / print statements
    │
    └─ SLOW PATH (line-by-line, if AST fails):
        ├─ Filter out invalid lines
        └─ Fix indentation
```

**`modify_lib(file_content, new_import)`** — patches the execution context:

Replaces `import numpy as np` in the `code_context` string (from the JSONL dataset) with the counterfactual import (e.g., `import WrapV2Numpy as np`). Returns `""` if the pattern is not found.

**`normalize_object_attributes(code, ast_module)`** — called after code cleaning in injection mode:

Invokes the appropriate `ast_cleaning_*.py` module to rewrite object attribute accesses before double-execution with real NumPy (e.g., strips `_v2` from `A.shape_v2`).

### 4.4 Counterfactual Wrappers — `Wrap*.py`

Each wrapper is a Python module that **intercepts NumPy attribute access** via a proxy class and routes it to the real `numpy` module after stripping/adjusting the name.

| File | Scheme | Accepts | Rejects |
|------|--------|---------|---------|
| `WrapV2Numpy.py` | `_v2` suffix | `np.mean_v2()` | `np.mean()` |
| `WrapUnderscoreNumpy.py` | trailing `_` | `np.mean_()` | `np.mean()` |
| `WrapCapitalizeNumpy.py` | Capital first letter | `np.Mean()` | `np.mean()` |

These modules are imported at runtime by swapping the `import numpy as np` line in the execution context (via `modify_lib()`). The `custom_lib_path` in the config tells Python where to find them.

### 4.5 AST Normalizers — `ast_cleaning_*.py`

Used exclusively during the **control_passed** double-execution step: the LLM code (written for the fake API) must be adapted back so it runs on real NumPy.

Each module exposes a `normalize(tree)` function that transforms the Python AST:

| File | Transformation | Exception |
|------|---------------|-----------|
| `ast_cleaning_v2.py` | `obj.attr_v2` → `obj.attr` | Allows `np.*` unchanged |
| `ast_cleaning_underscore.py` | `obj.attr_` → `obj.attr` | Allows `np.*` unchanged |
| `ast_cleaning_capitalize.py` | `obj.Attr` → `obj.attr` | `.T` stays uppercase |

If an attribute does **not** conform to the expected scheme (e.g., `obj.shape` in `_v2` mode), the module raises a `MODULE_WITH_SUFFIX_ERROR` — this prevents silently wrong normalizations and is recorded as an error in the results.

---

## 5. Documentation Generation — `src/documentation/`

### 5.1 Real NumPy Documentation

**`numpy_documentation_creation.py`**

Generates documentation files for real NumPy by introspecting the installed `numpy` module. Outputs:

- `real_minimal_numpy.txt` — function signatures + one-line descriptions
- `real_ultra_minimal_numpy.txt` — signatures only
- `real_full_doc_numpy.txt` — full docstrings

Run from `src/documentation/`:

```bash
python3 numpy_documentation_creation.py
```

### 5.2 Counterfactual Documentation Generators

Three scripts mirror `numpy_documentation_creation.py` but rename every function according to their scheme:

| Script | Output files | Example entry |
|--------|-------------|---------------|
| `v2_numpy_documentation.py` | `corrupted_*_numpy_v2.txt` | `numpy.mean_v2(a, axis=None, ...)` |
| `underscore_numpy_documentation.py` | `corrupted_*_numpy_underscore*.txt` | `numpy.mean_(a, axis=None, ...)` |
| `capitalize_numpy_documentation.py` | `corrupted_*_numpy_capitalize*.txt` | `numpy.Mean(a, axis=None, ...)` |

Each script generates three verbosity levels (`full`, `minimal`, `ultra_minimal`).

Run from `src/documentation/` (or `src/`):

```bash
python3 documentation/v2_numpy_documentation.py
python3 documentation/underscore_numpy_documentation.py
python3 documentation/capitalize_numpy_documentation.py
```

### 5.3 Noisy Documentation Generator

**`noisy_doc_generator.py`** — the most important documentation tool for the noise experiments.

**Concept**: real NumPy has hundreds of functions. The LLM has seen all of them during pre-training. If the documentation only contains the 30–40 functions actually used in DS-1000 (the "interest functions"), the LLM cannot miss the renaming. But when the documentation grows to include hundreds of *unchanged* functions alongside the renamed ones, the LLM may overlook the modifications — this is the "noise" effect.

**Parameters**:

| Noise level | Meaning |
|-------------|---------|
| 0% | Only renamed functions that appear in DS-1000 exercises |
| 25% | + 25% of other NumPy functions (unmodified) |
| 50% | + 50% of other NumPy functions |
| 75% | + 75% of other NumPy functions |
| 100% | All NumPy functions (+ renamed interest functions) |

**Inputs**:
- `interest_functions_v2.txt` / `_underscore.txt` / `_capitalize.txt` — the renamed functions that actually appear in DS-1000 prompts, reference code, and past LLM outputs
- The real NumPy introspection (via `documentation_dictionnary.py`)

**Output files** (in `src/documentation/`):

```
corrupted_{full,minimal,ultra_minimal}_numpy_{v2,underscore,capitalize}_noise{0,25,50,75,100}.txt
```

Run from `src/documentation/`:

```bash
python3 noisy_doc_generator.py
```

### 5.4 Generated Documentation Files

Complete inventory of documentation files in `src/documentation/`:

```
Real NumPy:
  real_minimal_numpy.txt
  real_ultra_minimal_numpy.txt
  real_full_doc_numpy.txt
  real_doc_numpy.txt

Counterfactual (no noise):
  corrupted_minimal_numpy_v2.txt
  corrupted_minimal_numpy_capitalize.txt
  corrupted_minimal_numpy.txt              ← underscore
  corrupted_ultra_minimal_numpy_v2.txt
  corrupted_ultra_minimal_numpy_capitalize.txt
  corrupted_ultra_minimal_numpy.txt        ← underscore
  corrupted_full_doc_numpy_v2.txt
  corrupted_full_doc_numpy_capitalize.txt
  corrupted_full_doc_numpy.txt             ← underscore

Noisy (5 noise levels × 3 verbosities × 3 schemes = 45 files):
  corrupted_{full,minimal,ultra_minimal}_numpy_{v2,underscore,capitalize}_noise{0,25,50,75,100}.txt

Interest function lists:
  interest_functions_v2.txt
  interest_functions_underscore.txt
  interest_functions_capitalize.txt

Textual explanations (injected as documentation variant):
  doc_explanation.txt
  doc_explanation_v2.txt
  doc_explanation_capitalize.txt
```

---

## 6. Dataset & Data Files — `data/`

### 6.1 Dataset Format (DS-1000 JSONL)

All datasets are **JSONL** files (one JSON object per line). Each object represents one coding problem:

```json
{
  "prompt": "Given a 1D NumPy array x with possible NaN values, remove all NaNs...",
  "reference_code": "x = x[~np.isnan(x)]",
  "metadata": {
    "problem_id": 300,
    "library_problem_id": 9,
    "library": "Numpy",
    "test_case_cnt": 2,
    "perturbation_type": "Origin",
    "perturbation_origin_id": 9
  },
  "code_context": "def test_execution(solution_code):\n    x = np.array([1.0, np.nan, 2.0])\n    exec(solution_code)\n    assert ...\n"
}
```

| Field | Role |
|-------|------|
| `prompt` | Sent verbatim to the LLM |
| `reference_code` | Ground truth (never shown to LLM; useful for inspection) |
| `metadata` | Problem identifiers and DS-1000 classification |
| `code_context` | Python test harness; the LLM's solution is injected and executed inside it |

**Key datasets**:

| File | Description | Tasks |
|------|-------------|-------|
| `ds1000_npyOnly.jsonl` | NumPy-only original | 159 |
| `ds1000_npyOnly_corrupted_v2.jsonl` | `_v2` renamed prompts + reference | 159 |
| `ds1000_npyOnly_corrupted_underscore.jsonl` | `_` renamed | 159 |
| `ds1000_npyOnly_corrupted_capitalize.jsonl` | Capitalized | 159 |
| `data_smallsample.jsonl` | Small subset for quick tests | ~10 |
| `ds1000.jsonl` | Full DS-1000 (all libraries) | ~1000 |

### 6.2 Dataset Corruption Scripts

Each script applies a renaming transformation to function names in `prompt`, `reference_code`, and `code_context`:

```bash
python3 data/data_modif_v2.py           # Appends _v2 to all np.* calls
python3 data/data_modif_underscore.py   # Appends _ to all np.* calls
python3 data/data_modif_capitalize.py   # Capitalizes function names
```

`data_extraction.py` — extracts the NumPy-only subset from the full `ds1000.jsonl`.

---

## 7. Configuration Files — `configs/`

### 7.1 Full Configuration Schema

```json
{
  "exec": {
    "timeout": 120,      // Seconds per code execution (subprocess timeout)
    "pass_at": 10        // Max LLM attempts per task; first success wins
  },

  "llm": {
    "model": ["qwen2.5-coder:32b", "codestral"],  // Models to benchmark
    "temperature": 0.9,   // Scalar OR {"model_name": value, "default": value}
    "seed": 42,           // Scalar OR {"model_name": value, "default": null}
                          // Incremented by 1 per pass@k attempt
    "num_ctx": 30000,     // Ollama context window (tokens)
    "api_url": "http://localhost:11434/api"
  },

  "data": {
    // Paths relative to src/ (where execution_process.py runs)
    "origin_data": "../data/ds1000_npyOnly.jsonl",
    "corrupted_data": "../data/ds1000_npyOnly_corrupted_v2.jsonl"
  },

  "real_lib": {
    "name": "Numpy",
    "custom_lib_path": null,          // null = use system numpy
    "system_prompt": "...",
    "documentation": {
      "nothing":      {"intro": "", "path": ""},
      "minimal":      {"intro": "Here is a minimal ...", "path": "/abs/path/real_minimal_numpy.txt"},
      "ultra_minimal":{"intro": "...", "path": "/abs/path/real_ultra_minimal_numpy.txt"}
    }
  },

  "new_lib_injection": {
    "name": "WrapV2Numpy",            // Module name (must be importable from custom_lib_path)
    "ast_cleaning_module": "ast_cleaning_v2",  // Module for AST normalization
    "custom_lib_path": "/abs/path/to/src",     // Added to sys.path at runtime
    "system_prompt": "...",
    "documentation": {
      "minimal":           {"intro": "...", "path": "/abs/.../corrupted_minimal_numpy_v2.txt"},
      "ultra_minimal":     {"intro": "...", "path": "/abs/.../corrupted_ultra_minimal_numpy_v2.txt"},
      "minimal_noise0":    {"intro": "...", "path": "/abs/.../corrupted_minimal_numpy_v2_noise0.txt"},
      "minimal_noise25":   {"intro": "...", "path": "/abs/.../corrupted_minimal_numpy_v2_noise25.txt"},
      "minimal_noise50":   {"intro": "...", "path": "/abs/.../corrupted_minimal_numpy_v2_noise50.txt"},
      "minimal_noise75":   {"intro": "...", "path": "/abs/.../corrupted_minimal_numpy_v2_noise75.txt"},
      "minimal_noise100":  {"intro": "...", "path": "/abs/.../corrupted_minimal_numpy_v2_noise100.txt"}
    }
  }
}
```

**Important**: all `path` values must be **absolute paths** (or paths valid from the working directory where the script is launched).

### 7.2 Available Configs

| File | Perturbation | Notes |
|------|-------------|-------|
| `config_bigexec_v2.json` | `_v2` | Full multi-model run, no noise |
| `config_bigexec_underscore.json` | `_` | Full multi-model run |
| `config_bigexec_capitalize.json` | Capitalize | Full multi-model run |
| `config_v2_explanation.json` | `_v2` | Uses `doc_explanation` as documentation |
| `config_underscore_explanation.json` | `_` | Explanation doc variant |
| `config_capitalize_explanation.json` | Capitalize | Explanation doc variant |

---

## 8. Running the Pipeline

### 8.1 Prerequisites

1. Python 3.10+ environment with dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama** running locally with required models already pulled:
   ```bash
   ollama pull qwen2.5-coder:32b
   ollama pull codestral
   # etc. — see OLLAMA_MODELS_REFERENCE.md
   ```

3. Verify Ollama is reachable at `http://localhost:11434` (the pipeline warm-up will wait for it).

### 8.2 Basic Execution

Always run from the **project root**:

```bash
python3 src/execution_process.py configs/config_bigexec_v2.json
```

This will:
- Create `results/run_YYYY-MM-DD_HH-MM-SS/`
- Save `config.json` snapshot there
- Run control mode + injection mode for every `(model, doc)` pair
- Write `results.jsonl` incrementally (safe to interrupt and resume)

### 8.3 Filtering Options

```bash
# Resume from a specific task (skips problem_id <= N)
python3 src/execution_process.py configs/config_bigexec_v2.json -t 150

# Only run one model
python3 src/execution_process.py configs/config_bigexec_v2.json --model qwen2.5-coder:32b

# Only run one documentation condition
python3 src/execution_process.py configs/config_bigexec_v2.json --doc minimal_noise50

# Skip injection, only run control baseline
python3 src/execution_process.py configs/config_bigexec_v2.json --control_only

# Skip control, only run injection
python3 src/execution_process.py configs/config_bigexec_v2.json --injection_only
```

### 8.4 SLURM / HPC Execution

`trick.sbatch` wraps the pipeline for SLURM clusters:

```bash
sbatch trick.sbatch configs/config_bigexec_v2.json
```

The script:
1. Sets up the Ollama server on GPU 0
2. Waits for Ollama to be ready
3. Pre-loads all models into GPU memory
4. Runs `execution_process.py` with stdout redirected to `logs_slurm/job_{SLURM_JOB_ID}.out`
5. Cleans up Ollama on exit

---

## 9. Results Format

### Directory structure

```
results/
└── run_2026-04-05_14-30-00/
    ├── config.json       ← exact config used (for reproducibility)
    └── results.jsonl     ← one JSON line per task
```

### `results.jsonl` schema

**Injection mode entry** (most complete):

```json
{
  "task_id": 300,
  "metadata": {
    "problem_id": 300,
    "library_problem_id": 9,
    "library": "Numpy",
    "test_case_cnt": 2,
    "perturbation_type": "Origin",
    "model_name": "qwen2.5-coder:32b",
    "doc_name": "minimal_noise25",
    "mode": "injection",
    "temperature": 0.9,
    "seed": 42,
    "token_count": 5432,
    "pass_at": 10,
    "pass_at_attempt": 3,
    "pass_at_success": true
  },
  "passed": true,
  "control_passed": false,
  "llm_code": "x = x[~np.isnan_v2(x)]",
  "stdout": "SUCCESS_MARKER\n",
  "stderr": "",
  "stdout_control": "EXEC_ERROR: module 'numpy' has no attribute 'isnan_v2'\n",
  "stderr_control": "Traceback (most recent call last): ...",
  "full_response": "```python\nx = x[~np.isnan_v2(x)]\n```",
  "is_control": false
}
```

**Control mode entry** (simpler):

```json
{
  "task_id": 300,
  "metadata": { "...", "mode": "control", "is_control": true },
  "passed": true,
  "llm_code": "x = x[~np.isnan(x)]",
  "stdout": "SUCCESS_MARKER\n",
  "stderr": "",
  "is_control": true
}
```

**Error entry** (LLM API failure):

```json
{
  "task_id": 300,
  "metadata": { "..." },
  "error": "LLM_API_FAILURE",
  "passed": false,
  "control_passed": false,
  "is_control": false
}
```

| Field | Description |
|-------|-------------|
| `passed` | Code succeeded in the current mode (fake lib for injection, real numpy for control) |
| `control_passed` | Same code also succeeded with real NumPy (injection only) |
| `llm_code` | Final cleaned code that was executed |
| `full_response` | Raw LLM output (including markdown fences) |
| `pass_at_attempt` | Which attempt (1-indexed) produced this result |
| `token_count` | Input token count (from Ollama or estimated) |
| `error` | Set only on failure: `LLM_API_FAILURE`, `SYNTAX_ERROR`, `MODULE_WITH_SUFFIX_ERROR`, `MISSING_CONTEXT_IN_DATASET` |

---

## 10. Analysis & Visualization — `src/perf_review/`

Run all analysis scripts from the **project root**.

### 10.1 `plot_model_perf.py`

Generates a grouped bar chart comparing control vs injection performance across models and documentation conditions.

```bash
python3 src/perf_review/plot_model_perf.py <results.jsonl> -o <output_dir>
```

**What it plots** (one group of bars per model):

| Bar color/label | Source rows | Metric |
|----------------|-------------|--------|
| Control classique | `is_control=True`, `doc_name="nothing"` | `passed` |
| Control minimal | `is_control=True`, `doc_name="minimal"` | `passed` |
| Control ultra_minimal | `is_control=True`, `doc_name="ultra_minimal"` | `passed` |
| Doc minimal | `is_control=False`, `doc_name="minimal"` | `passed` |
| Doc ultra_minimal | `is_control=False`, `doc_name="ultra_minimal"` | `passed` |
| Doc minimal (orig. eval) | `is_control=False`, `doc_name="minimal"` | `control_passed` |
| Doc ultra_minimal (orig. eval) | `is_control=False`, `doc_name="ultra_minimal"` | `control_passed` |

**Output**: `plot_global_control_vs_doc_all.png`

### 10.2 `plot_noisy_doc_perf.py`

Generates histograms showing performance as a function of documentation noise level.

```bash
python3 src/perf_review/plot_noisy_doc_perf.py <results.jsonl> -o <output_dir>
```

**What it plots**:
- X-axis: noise levels (0%, 25%, 50%, 75%, 100%)
- Y-axis: success rate (%)
- One bar per `(model, verbosity)` combination
- Bar labels show `successes / total` counts
- Separate panels or grouped bars for `passed` vs `control_passed`

**Output**: `plot_noisy_doc_perf.png`

### 10.3 `sanity_check.py`

Produces a human-readable quality report detecting anomalies in the results.

```bash
python3 src/perf_review/sanity_check.py <results.jsonl> [-o report.txt]
```

**Checks performed**:

| Check | Description |
|-------|-------------|
| `LLM_API_FAILURE` | Count and list tasks where the LLM never responded |
| `both_passed` | `passed=True AND control_passed=True` — LLM may not have used the fake API |
| `no_np_passed` | Tasks that passed without any `np.*` call in the code |
| `injection_pass_control_fail` | `passed=True, control_passed=False` — strong evidence of doc-following (the desired signal) |
| `perturbation_adoption` | Did the LLM actually include the suffix/capitalization in its code? |
| Global pass rates | Breakdown by `(model, doc_name, mode)` |

**Output**: prints to stdout + optional `report.txt`

### 10.4 `advanced_sanity_check.py`

Additional diagnostic utilities for deeper investigation. Includes error pattern analysis and per-task breakdowns. Run the same way as `sanity_check.py`.

---

## 11. Interpreting the Metrics

The pipeline produces two independent binary metrics per injection task:

```
passed           = did the LLM code work with the counterfactual library?
control_passed   = did the SAME code work with real NumPy?
```

**Reading a results file for research conclusions**:

1. **High `passed`, low `control_passed`**: LLMs are successfully following the injected documentation and have genuinely abandoned their parametric knowledge for those tasks. This is the strongest evidence that the model reads the context.

2. **High `both_passed` rate**: The modified functions in those tasks may be used on `ndarray` objects (e.g., `A.mean()`) where the AST normalization strips the suffix, making both executions equivalent. Use `sanity_check.py` to identify and filter these.

3. **Noise effect**: Comparing `passed` across noise levels (0% → 100%) shows how quickly models lose track of the renamed functions when drowned in unmodified documentation.

4. **Explanation doc**: Using `doc_explanation_*.txt` (a short textual paragraph explaining "the library is the same as NumPy but with `_v2` added") instead of the full API listing tests whether a brief conceptual description suffices.

---

## 12. Creating a New Experiment

**Step 1 — Choose a perturbation scheme**: `v2`, `underscore`, or `capitalize`.

**Step 2 — Generate documentation** (if not already done):
```bash
cd src/documentation
python3 v2_numpy_documentation.py        # or underscore/capitalize variant
python3 noisy_doc_generator.py           # generates all noise levels
```

**Step 3 — Prepare the dataset** (if not already done):
```bash
python3 data/data_modif_v2.py            # or underscore/capitalize
```

**Step 4 — Create a config** by copying the closest existing one:
```bash
cp configs/config_bigexec_v2.json configs/my_experiment.json
```

Edit the copy:
- `llm.model` — list of Ollama model names you have available
- `data.corrupted_data` — path to the matching corrupted JSONL
- `new_lib_injection.name` — wrapper module name
- `new_lib_injection.ast_cleaning_module` — matching normalizer
- `new_lib_injection.documentation` — paths to the doc files you generated
- `exec.pass_at` — number of LLM retries per task
- `llm.temperature` and `llm.seed` — sampling parameters

**Step 5 — Run**:
```bash
python3 src/execution_process.py configs/my_experiment.json
# or on SLURM:
sbatch trick.sbatch configs/my_experiment.json
```

**Step 6 — Analyze**:
```bash
RESULTS=results/run_YYYY-MM-DD_HH-MM-SS

python3 src/perf_review/plot_model_perf.py      $RESULTS/results.jsonl -o $RESULTS
python3 src/perf_review/plot_noisy_doc_perf.py  $RESULTS/results.jsonl -o $RESULTS
python3 src/perf_review/sanity_check.py         $RESULTS/results.jsonl
```
