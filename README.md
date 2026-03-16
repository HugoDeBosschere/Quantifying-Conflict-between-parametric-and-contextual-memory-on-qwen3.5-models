## Mémoire paramétrique vs mémoire factuelle des LLM (projet DS-1000 + Numpy)

Ce dépôt contient une pipeline complète pour évaluer **dans quelle mesure un LLM s’appuie sur sa mémoire paramétrique** (ce qu’il “sait” déjà de Numpy) **vs la documentation fournie en contexte** (mémoire factuelle / RAG).

L’idée centrale :
- **On crée des “librairies contrefactuelles”** (wrappers Numpy : renommage des fonctions, suffixes `_v2`, `_`, etc.).
- On fournit au LLM une **documentation cohérente avec cette fausse librairie**.
- On lui donne des **problèmes DS-1000** (Numpy) avec un moteur de tests.
- On exécute la solution du LLM dans deux mondes :
  - **Injection** : avec la librairie contrefactuelle (ex. `WrapV2Numpy`).
  - **Control** : avec la vraie librairie Numpy.
- On mesure :
  - `passed` : succès avec la librairie vue par le LLM (contrefactuelle).
  - `control_passed` : succès de la même solution avec la **vraie** librairie Numpy.

Cela permet de voir si le LLM suit réellement la doc contrefactuelle (erreurs avec la vraie lib) ou s’il “se rappelle” la vraie API.

---

## Vue d’ensemble de la pipeline

- **Données** : DS-1000, restreint à Numpy (`../data/ds1000_npyOnly*.jsonl`).
- **Librairies** :
  - `real_lib` : Numpy classique.
  - `new_lib_injection` : wrappers contrefactuels (`WrapV2Numpy`, `WrapUnderscoreNumpy`, `WrapCapitalizeNumpy`, etc.) dans `src/`.
- **LLM** :
  - Client générique via Ollama (`src/llmclient.py`).
  - Modèle(s) configurés dans la section `llm` du fichier de config.
- **Exécution** :
  - Orchestrée par `src/execution_process.py` à partir d’un fichier de config JSON.
  - Pour chaque `(modèle, documentation)` :
    - **Mode control** : vraie doc Numpy (contrôle de base).
    - **Mode injection** : doc contrefactuelle + wrapper.
- **Résultats** :
  - Un répertoire `results/run_YYYY-MM-DD_HH-MM-SS/` par exécution.
  - Un snapshot de la config (`config.json`) + un `results.jsonl` (une ligne par tâche).
- **Analyse / plots** :
  - `src/perf_review/plot_model_perf.py` : comparaison globale control vs injection (+ métrique `control_passed`).
  - `src/perf_review/plot_noisy_doc_perf.py` : perfs en fonction du **bruit dans la documentation** (docs “noisy”).
  - `src/perf_review/sanity_check.py` : contrôles qualité et anomalies.

---

## Composants principaux

- **`src/execution_process.py`**
  - Charge la config JSON (`load_config_from_path`).
  - Crée un dossier de run dans `results/` et y sauvegarde la config (`setup_run_directory`).
  - Pour chaque modèle/config de doc :
    - Mode **control** (`run_control`) : lit `data.origin_data` (DS-1000 d’origine), interroge le LLM avec la vraie doc Numpy, exécute le code et écrit les résultats (avec `is_control=True`).
    - Mode **injection** (`run_benchmark`) : lit `data.corrupted_data` (dataset modifié), interroge le LLM avec la doc contrefactuelle, exécute d’abord avec la **lib contrefactuelle**, puis avec la **vraie lib** pour remplir `control_passed`.

- **`src/llmclient.py`**
  - Gère la communication avec Ollama (`/api/generate`, éventuellement `/api/tokenize`).
  - Construit le **system prompt** et le **contexte** (documentation injectée) à partir de la config :
    - `real_lib` ou `new_lib_injection` selon `mode` (`control` / `injection`).
    - Chargement des fichiers de doc (intro + texte).
  - Gère un **warm-up** robuste (attente que le serveur soit prêt, requête minimale “Say OK.”).

- **`src/cleaning.py`**
  - Nettoie les réponses du LLM avant exécution :
    - Extraction du code à l’intérieur des balises ``` ou `<code>`.
    - Nettoyage AST : suppression des imports, transformation des `return` en `result = ...`, filtrage des blocs statiques inutiles, etc.
    - Gestion de l’indentation / lignes “bruitées”.
  - `modify_lib` : remplace l’import `import numpy as np` dans le contexte d’exécution par l’import de la lib contrefactuelle (ex. `import WrapV2Numpy as np`).

- **Wrappers Numpy (`src/Wrap*.py`)**
  - Modules Python qui exposent une API Numpy modifiée :
    - Suffixe `_v2` (`WrapV2Numpy`), underscore (`WrapUnderscoreNumpy`), capitalisation (`WrapCapitalizeNumpy`), etc.
  - La doc contrefactuelle explique comment utiliser ces fonctions.

- **Scripts d’analyse (`src/perf_review/*.py`)**
  - `plot_model_perf.py` :
    - Regroupe les résultats par **modèle** et par **type de documentation** :
      - `Control classique`, `Control minimal`, `Control ultra_minimal` (runs control).
      - `Doc minimal`, `Doc ultra_minimal`, `Doc explanation` (runs injection).
      - `Doc minimal (éval. lib d'origine)`, etc. (métrique `control_passed` sur les runs injection).
  - `plot_noisy_doc_perf.py` :
    - Histogrammes des taux de réussite par doc “noisy” (e.g. `minimal_noise0`, `ultra_minimal_noise75`, etc.).
    - Une barre par `(modèle, doc)`, texte `success/total` au-dessus.
  - `sanity_check.py` :
    - Sections de diagnostic (LLM_API_FAILURE, anomalies où `passed=True` mais `control_passed=False`, etc.).

---

## Format des fichiers de configuration

Les configs sont des **fichiers JSON** (malgré le terme “config.yaml” dans certaines discussions). Exemples :
- `config_bigexec_v2_noisy_comparison.json`
- `config_bigexec_capitalize_noisy_comparison.json`
- `config_bigexec_underscore_noisy_comparison.json`

### Structure générale

- **`exec`**
  - `timeout` : timeout (en secondes) pour l’exécution du code Python de la solution (dans `execute_task_engine`).

- **`llm`**
  - `model` : liste de noms de modèles à tester (`["codestral"]`, `["qwen2.5-coder:32b", "gemma3:12b", ...]`).
  - `temperature` : température passée à Ollama.
  - `num_ctx` : longueur de contexte.
  - `api_url` : URL de l’API Ollama (`http://localhost:11434/api`).
  - Champs `_comment_*` : commentaires internes (ignorés par le code).

- **`data`**
  - `origin_data` : chemin **relatif à `src/`** du dataset DS-1000 d’origine (ex. `"../data/ds1000_npyOnly.jsonl"`).
  - `corrupted_data` : chemin **relatif à `src/`** du dataset contrefactuel (ex. `"../data/ds1000_npyOnly_corrupted_v2.jsonl"` ou `"../data/ds1000_npyOnly_corrupted_.jsonl"` pour underscore).

- **`real_lib`** (librairie “vraie” / contrôle)
  - `name` : `"Numpy"`.
  - `custom_lib_path` : `null` (on utilise Numpy du venv système).
  - `system_prompt` : instructions générales pour le LLM en mode contrôle :
    - Ne pas recréer les données.
    - Ne pas importer de libs standards.
    - Se limiter à des appels `np.*`.
    - Retourner uniquement du code Python exécutable dans un bloc markdown.
  - `documentation` : dictionnaire `{doc_name -> {intro, path}}` :
    - `nothing` : pas de doc.
    - `minimal` : chemin vers doc Numpy minimale.
    - `ultra_minimal` : doc ultra condensée.

- **`new_lib_injection`** (librairie contrefactuelle)
  - `name` : nom du module wrapper (`"WrapV2Numpy"`, `"WrapUnderscoreNumpy"`, …).
  - `custom_lib_path` : chemin absolu de `src` (pour que Python trouve le module wrapper).
  - `system_prompt` : similaire à `real_lib` mais adapté à l’API contrefactuelle :
    - Ex. “les fonctions doivent s’appeler `np.fonction_v2()`” ou `np.fonction_()`.
  - `documentation` : dictionnaire `{doc_name -> {intro, path}}` pour les docs contrefactuelles :
    - Cases simples : `minimal`, `ultra_minimal`.
    - Cas “noisy” : `minimal_noise0`, `minimal_noise25`, …, `ultra_minimal_noise100` pointant vers
      `src/documentation/noisy_doc/corrupted_*_numpy_{v2,underscore,capitalize}_noiseXX.txt`.

### Bonnes pratiques pour créer une nouvelle config

1. **Copier** une config existante la plus proche de votre scénario (ex. `config_bigexec_v2_noisy_comparison.json`).
2. Adapter :
   - `llm.model` : les modèles que vous avez dans Ollama.
   - `data.origin_data` / `data.corrupted_data` : vos fichiers DS-1000 filtrés/modifiés.
   - `new_lib_injection.name` + `custom_lib_path` : module wrapper et chemin vers `src`.
   - `system_prompt` de `new_lib_injection` : bien spécifier la contrainte sur les noms de fonctions.
   - `documentation` (control + injection) : pointeurs vers les bons fichiers de doc (réelle ou bruitée).
3. Vérifier que **tous les chemins sont valides** depuis `src/` (le script d’exécution est lancé depuis là).

---

## Comment lancer une exécution

### 1. Pré-requis

- Python 3.x + venv avec les dépendances (Numpy, Matplotlib, Requests, etc.).
- Serveur **Ollama** en route, avec les modèles nécessaires déjà téléchargés (noms utilisés dans `llm.model`).

### 2. Lancement simple (depuis la racine du projet)

Exécution directe en Python :

```bash
python3 src/execution_process.py config_bigexec_v2_noisy_comparison.json
```

Ou, pour une config underscore par exemple :

```bash
python3 src/execution_process.py config_bigexec_underscore_noisy_comparison.json
```

Le script :
- charge la config,
- crée un dossier `results/run_YYYY-MM-DD_HH-MM-SS/`,
- lance pour chaque modèle + chaque documentation :
  - le **mode control** (sauf si `--injection_only`) ;
  - le **mode injection** (sauf si `--control_only`).

### 3. Options de relance / filtrage

`src/execution_process.py` accepte plusieurs options pratiques :

- **Reprendre à partir d’un task_id** :
  ```bash
  python3 src/execution_process.py config_bigexec_v2_noisy_comparison.json -t 500
  ```
  → ignore les tâches avec `problem_id <= 500`.

- **Filtrer sur un modèle** :
  ```bash
  python3 src/execution_process.py config_bigexec_v2_noisy_comparison.json --model codestral
  ```

- **Filtrer sur une documentation spécifique** (clé dans `documentation`) :
  ```bash
  python3 src/execution_process.py config_bigexec_v2_noisy_comparison.json --doc minimal_noise50
  ```

- **Ne lancer que le mode control** :
  ```bash
  python3 src/execution_process.py config_bigexec_v2_noisy_comparison.json --control_only
  ```

- **Ne lancer que le mode injection** :
  ```bash
  python3 src/execution_process.py config_bigexec_v2_noisy_comparison.json --injection_only
  ```

Sur cluster (SLURM), le projet utilise un script `trick.sbatch` qui wrappe simplement cette commande en job batch, par exemple :

```bash
sbatch trick.sbatch config_bigexec_v2_noisy_comparison.json
```

---

## Où sont stockés les résultats et sous quel format ?

### Organisation dans `results/`

Chaque exécution crée un sous-dossier :

- `results/run_YYYY-MM-DD_HH-MM-SS/`
  - `config.json` : **copie exacte** de la config utilisée.
  - `results.jsonl` : une ligne JSON par tâche évaluée.
  - fichiers PNG générés par les scripts `perf_review` (plots).

Pour les expériences regroupées ou renommées, vous trouverez aussi des dossiers comme :
- `results/qwen_noisy_v2/`
- `results/qwen_noisy_capitalize/`
- `results/qwen_gemma_codestral_devstral_v2_docs/`
avec à chaque fois :
- un `config.json` (snapshot),
- un `results.jsonl`,
- des plots (e.g. `plot_global_control_vs_doc_all.png`, `plot_noisy_doc_perf.png`, etc.).

### Schéma d’une ligne de `results.jsonl`

Une ligne typique (control) :

- **Clés de haut niveau** :
  - `task_id` : identifiant de la tâche (souvent `problem_id` DS-1000).
  - `metadata` : dictionnaire avec :
    - `problem_id`, `library_problem_id`, `library` (souvent `"Numpy"`),
    - `test_case_cnt` : nombre de tests unitaires,
    - `perturbation_type` : `Origin`, `Semantic`, `Surface`, `Difficult-Rewrite`, etc.,
    - `perturbation_origin_id` : id DS-1000,
    - `model_name` : nom du modèle (ex. `"qwen2.5-coder:32b"`),
    - `doc_name` : clé de doc utilisée (`nothing`, `minimal`, `minimal_noise25`, …),
    - `mode` : `"control"` ou `"injection"`,
    - `temperature` : température du LLM,
    - `token_count` : décompte de tokens (soit fourni par Ollama, soit estimé).
  - `passed` : booléen, **succès des tests dans le mode courant** :
    - en mode control : succès avec la vraie Numpy (baseline).
    - en mode injection : succès avec la librairie contrefactuelle.
  - `control_passed` (seulement pour injection) :
    - booléen, succès de **la même solution** mais évaluée avec la vraie librairie d’origine.
  - `llm_code` : code final nettoyé (celui exécuté).
  - `stdout`, `stderr` : sorties de l’exécution dans le mode courant.
  - `stdout_control`, `stderr_control` : sorties de l’exécution de contrôle (pour injection uniquement).
  - `full_response` : réponse brute du LLM (incluant souvent le markdown).
  - `is_control` : `true` pour les runs control, `false` pour injection.

En cas d’échec de l’API LLM :
- `error` : `"LLM_API_FAILURE"`.
- `passed` : `false`, `control_passed` : `false`.
- `metadata` est **toujours rempli** (modèle, doc, etc.), ce qui permet des sanity-checks robustes.

---

## Analyse et visualisation des résultats

Depuis la racine du projet :

- **Plot global control vs injection + control_passed** :

```bash
python3 src/perf_review/plot_model_perf.py results/run_YYYY-MM-DD_HH-MM-SS/results.jsonl -o results/run_YYYY-MM-DD_HH-MM-SS
```

- **Plot spécifique docs bruitées** :

```bash
python3 src/perf_review/plot_noisy_doc_perf.py results/qwen_noisy_v2/results.jsonl -o results/qwen_noisy_v2
```

- **Sanity check complet** (qualité des données, anomalies) :

```bash
python3 src/perf_review/sanity_check.py results/run_YYYY-MM-DD_HH-MM-SS/results.jsonl
```

Les scripts produisent des fichiers PNG prêts à être insérés dans un rapport.

---

## Pour un·e nouveau·elle contributeur·rice : comment lancer sa première expérience ?

1. **Cloner le projet** et créer un environnement Python avec les dépendances.
2. Vérifier que **Ollama tourne** et que les modèles voulus sont disponibles.
3. Choisir une config existante (par ex. `config_bigexec_v2_noisy_comparison.json`) et l’adapter si besoin :
   - chemins `origin_data` / `corrupted_data`,
   - wrappers (`WrapV2Numpy` / `WrapUnderscoreNumpy` / …),
   - fichiers de documentation (réelle / contrefactuelle / noisy).
4. Lancer :
   ```bash
   python3 src/execution_process.py config_bigexec_v2_noisy_comparison.json
   ```
   ou via SLURM :
   ```bash
   sbatch trick.sbatch config_bigexec_v2_noisy_comparison.json
   ```
5. Une fois le run terminé, explorer :
   - `results/run_.../config.json` (config exacte),
   - `results/run_.../results.jsonl`,
   - les PNG générés dans ce dossier ou dans `results/..._docs/`.
6. Utiliser les scripts `perf_review` pour générer les **graphiques de comparaison** et lire les rapports de sanity check.

Avec ces éléments, quelqu’un qui ne connaît pas le projet peut comprendre la logique de la pipeline, écrire son propre fichier de config, lancer une exécution et analyser les résultats sans toucher au code interne.

