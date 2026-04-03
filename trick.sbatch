#!/bin/bash
# Les fichiers job_%j.out / job_%j.err ne sont créés que si tu lances avec:
#   sbatch trick.sbatch
# Avec "srun --pty bash" tu as un nœud mais pas de fichiers de log Slurm (tout va dans le terminal).
#
#SBATCH --job-name=eval_ollama
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=36:00:00
#SBATCH --exclude=sh[00,10-19]
#SBATCH --output=/usr/users/sdim/sdim_17/PFE/memory_code_eval/logs_slurm/job_%j.out
#SBATCH --error=/usr/users/sdim/sdim_17/PFE/memory_code_eval/logs_slurm/job_%j.err

# --- 0. CONFIGURATION : CONFIG OBLIGATOIRE EN ARGUMENT ---
# Usage : sbatch trick.sbatch chemin/vers/config [optionnel: autre args pour le script Python]
# Ex.   : sbatch trick.sbatch src/config.json
#        sbatch trick.sbatch src/config_bigexec.json -t 42
if [ -z "$1" ]; then
  echo "Usage: sbatch trick.sbatch <chemin/vers/config> [-t TASK_ID]"
  echo "Ex.   sbatch trick.sbatch src/config.json"
  exit 1
fi

CONFIG_PATH="$1"
shift
# Les autres arguments ($@) sont passés au script Python (ex. -t 42)

# Chemin absolu vers ton dossier projet
PROJECT_DIR="/usr/users/sdim/sdim_17/PFE/memory_code_eval"

# Chemin absolu vers le dossier contenant l'exécutable ollama
OLLAMA_BIN_DIR="/usr/users/sdim/sdim_17/apps/ollama/bin"

# Dossier pour les logs (absolu pour que ça marche même si sbatch est lancé depuis ailleurs)
LOG_DIR="${PROJECT_DIR}/logs_slurm"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Date début   : $(date)"
echo "Noeud        : $(hostname)"
echo "=========================================="

# --- 1. ACTIVATION DE L'ENVIRONNEMENT PYTHON ---
echo "[1/4] Activation de l'environnement virtuel..."
module load python/3.11.2/gcc-13.1.0
source "${PROJECT_DIR}/.venv/bin/activate"

# --- 2. LANCEMENT DU SERVEUR OLLAMA ---
echo "[2/4] Démarrage du serveur Ollama..."

# On se déplace dans le dossier du binaire pour pouvoir faire ./ollama
cd "$OLLAMA_BIN_DIR"

# --- Choix GPU : un seul (pipeline) ou plusieurs (modèle lourd) ---
# Mode A — Pipeline inférence + exécution (modèle tient sur 1 GPU) : Ollama sur GPU 0 uniquement
export CUDA_VISIBLE_DEVICES=0
# Mode B — Modèle lourd réparti sur 2+ GPU : commenter la ligne ci-dessus (Ollama verra tous les GPU)
#          Optionnel : forcer la répartition sur tous les GPU visibles
# export OLLAMA_SCHED_SPREAD=1

export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_CONTEXT_LENGTH=30000
# Si tu veux changer l'endroit où les modèles sont stockés :
# export OLLAMA_MODELS="/usr/users/sdim/sdim_17/.ollama/models"

# Lancement en arrière-plan avec nohup
# > ollama.log 2>&1  : Redirige toutes les sorties d'Ollama vers un fichier log dédié
# &                  : Lance en tâche de fond (background)
nohup ./ollama serve > "${LOG_DIR}/ollama_server_${SLURM_JOB_ID}.log" 2>&1 &

# On récupère l'ID du processus pour le tuer plus tard
OLLAMA_PID=$!
echo "-> Serveur lancé avec PID: $OLLAMA_PID"
echo "-> Pause de 15 secondes pour laisser le temps au serveur de démarrer..."
sleep 15

# Petite vérification que le serveur répond (optionnel)
# On reste dans le dossier bin pour faire ./ollama list
echo "-> Vérification des modèles disponibles :"
./ollama list

# --- 3. EXECUTION DU SCRIPT PYTHON ---
echo "[3/4] Lancement du script d'évaluation..."

# On retourne dans le dossier du projet
cd "$PROJECT_DIR"


echo "♨️  Préchauffage des modèles (Loading into RAM)..."
MODELS=$(python -c "import json, sys; cfg=json.load(open(sys.argv[1])); print(' '.join(cfg['llm']['model']))" "$CONFIG_PATH")

for MODEL in $MODELS; do
  echo "-> Chargement de ${MODEL}..."
  curl -s http://localhost:11434/api/generate -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"Hi\",
    \"stream\": false
  }" > /dev/null
  echo "   ${MODEL} chargé."
done

echo "✅ Tous les modèles chargés ! Lancement du script Python (config: $CONFIG_PATH)..."
python -u src/execution_process.py "$CONFIG_PATH" "$@"

EXIT_CODE=$?

# --- 4. NETTOYAGE ---
echo "[4/4] Fin du script Python (Code: $EXIT_CODE). Arrêt d'Ollama..."
kill $OLLAMA_PID

echo "=========================================="
echo "Job terminé à $(date)"
echo "=========================================="

exit $EXIT_CODE