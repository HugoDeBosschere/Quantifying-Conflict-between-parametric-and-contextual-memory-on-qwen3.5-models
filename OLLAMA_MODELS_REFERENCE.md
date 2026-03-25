# Référence des modèles Ollama (en vogue)

Tableaux par catégorie : **espace disque** (ordre de grandeur), **fenêtre de contexte** (tokens), **spécificité**.  
Commande d’installation : `ollama pull <tag>`.

---

## Modèles code

| Modèle (tag) | Paramètres | Espace disque | Contexte | Spécificité |
|--------------|------------|---------------|----------|-------------|
| **qwen2.5-coder** | 0.5B–32B | 0.5B ~1 GB, 3B ~2 GB, 7B ~4.7 GB, 14B ~9 GB, 32B ~19–20 GB | 32K (128K selon variante) | Référence code open-source, très bon en génération/réparation, 32B proche GPT-4o |
| **qwen3-coder** / **qwen3-coder-next** | 3B–480B | 30B ~19 GB, 480B ~290 GB | jusqu’à 256K | Nouvelle génération, orienté agent / workflow |
| **deepseek-coder** | 1.3B, 6.7B, 33B | 6.7B ~4 GB, 33B ~16–19 GB | 16K | 2T tokens (87 % code), multilingue (EN/ZH) |
| **deepseek-coder-v2** | — | variable | long | Niveau “GPT-4 Turbo” sur tâches code (benchmarks) |
| **codellama** | 7B, 13B, 34B, 70B | 7B ~3.8 GB, 13B ~7.4 GB, 34B ~19 GB, 70B ~39 GB | 16K (70B: 2K) | Meta, base code, 34B/70B pour gros modèles |
| **codegemma** | 2B, 7B | ~1.6 GB, ~4 GB | 8K–16K | Google, code + inférence, léger |
| **codestral** | 22B | ~13 GB | 32K | Mistral, 80+ langages |
| **devstral** | 24B | ~14 GB | 128K | Mistral / All Hands, agents logiciels, édition multi-fichiers |
| **granite-code** | 3B, 8B, 20B, 34B | 2 GB, 4.6 GB, 12 GB, 19 GB | 3B/8B: 125K ; 20B/34B: 8K | IBM, plusieurs tailles, 3B/8B à long contexte |
| **starcoder2** | 15B | ~9 GB | 16K | BigCode, code polyglotte |
| **deepcoder** | — | variable | — | Orienté code / raisonnement |

---

## Modèles généralistes

| Modèle (tag) | Paramètres | Espace disque | Contexte | Spécificité |
|--------------|------------|---------------|----------|-------------|
| **llama3.1** | 8B, 70B, 405B | 8B ~4.9 GB, 70B ~43 GB, 405B ~243 GB | 128K | Meta, généraliste, 8B très utilisé |
| **llama3.2** | 1B, 3B | 1B ~1–3 GB, 3B ~2–7 GB | 128K | Meta, edge / petit déploiement |
| **llama3.3** | 70B | ~43 GB | 128K | Meta, 70B aligné sur les perfs 405B, math + instructions |
| **llama3.2-vision** | 11B, 90B | ~8 GB, ~50+ GB | 128K | Multimodal (texte + image) |
| **qwen2.5** | 0.5B–72B | 7B ~4.7 GB, 14B ~9 GB, 32B ~20 GB, 72B ~40+ GB | 32K | Alibaba, généraliste, multilingue |
| **qwen3** / **qwen3.5** | 4B–72B+ | variable | long (128K+) | Dernière génération Qwen |
| **gemma2** | 2B, 9B, 27B | 1.6 GB, 5.5 GB, 16 GB | 8K | Google, généraliste |
| **gemma3** | 1B, 4B, 12B, 27B | 1B ~0.8 GB, 4B ~3.3 GB, 12B ~8 GB, 27B ~17 GB | 1B: 32K ; 4B/12B/27B: 128K | Google, texte + image (4B+), 27B “flagship” 1 GPU |
| **mistral** | 7B | ~4.1 GB | 32K | Mistral AI, généraliste efficace |
| **mistral-small** | 22B | ~13 GB | 32K+ | Bon compromis taille/qualité |
| **mistral-large** / **mistral-large2** | 123B | ~73 GB | 128K | Très gros, raisonnement avancé |
| **mistral-nemo** | 12B | ~7 GB | 128K | Léger, long contexte |
| **phi3** | 3.8B, 14B | 3.8B ~2.3 GB, 14B ~7.9 GB | 4K–128K selon variante | Microsoft, petit et rapide |
| **phi4** / **phi4-mini** | 14B | ~9 GB | long | Microsoft, évolution de Phi |
| **mixtral** | 8x7B (MoE) | ~26 GB | 32K | Mistral, MoE, bon débit |
| **granite3.1-moe** / **granite4** | — | variable | long | IBM, MoE / généraliste |
| **dolphin3** / **dolphin-*** | dérivés Llama/Mistral | selon base | 128K+ | Fine-tune “uncensored”, chat |
| **yi** | 6B–34B | variable | 4K–128K | General-purpose, multilingue |
| **falcon3** | 8B–70B | variable | 8K+ | TII, généraliste |
| **olmo2** | 7B–32B | variable | — | Allen AI, open, reproduction |

---

## Modèles raisonnement / chaînage de pensée

| Modèle (tag) | Paramètres | Espace disque | Contexte | Spécificité |
|--------------|------------|---------------|----------|-------------|
| **deepseek-r1** | 1.5B–671B | 7B ~4.7 GB, 70B (distill) ~43 GB, 671B ~404 GB | long | Raisonnement step-by-step, “reasoning”, distillation Llama |
| **phi4-reasoning** | 14B | ~9 GB | long | Microsoft, orienté raisonnement |
| **deepseek-v3** | 671B | ~400+ GB | long | Généraliste + raisonnement, très gros |
| **openthinker** / **lfm2.5-thinking** | — | variable | — | Modèles “thinking” / raisonnement explicite |
| **cogito** | — | variable | — | Raisonnement / planification |

---

## Modèles légers / edge

| Modèle (tag) | Paramètres | Espace disque | Contexte | Spécificité |
|--------------|------------|---------------|----------|-------------|
| **tinyllama** | 1.1B | ~0.6 GB | 2K | Très léger, démo / edge |
| **smollm2** / **smollm** | 1.7B–3B | ~1–2 GB | 32K | Petit, long contexte |
| **llama3.2:1b** / **:3b** | 1B, 3B | 1–3 GB, 2–7 GB | 128K | Meta, edge |
| **phi3:mini** | 3.8B | ~2.3 GB | 4K–128K | Microsoft, faible VRAM |
| **qwen2.5:0.5b** / **:1.5b** / **:3b** | 0.5B–3B | <1–2 GB | 32K | Très petit, mobile / edge |
| **gemma3:1b** | 1B | ~0.8 GB | 32K | Google, edge |
| **minicpm-v** | — | petit | — | Vision léger |
| **ministral-3** | — | petit | — | Mistral, très léger |

---

## Embeddings (recherche / RAG)

| Modèle (tag) | Dimension | Espace disque | Spécificité |
|--------------|-----------|---------------|-------------|
| **nomic-embed-text** | 768 | ~0.3 GB | Texte, généraliste |
| **mxbai-embed-large** | 1024 | ~0.6 GB | Multilingue, retrieval |
| **bge-m3** | 1024 | ~1 GB | Multilingue, dense + sparse |
| **snowflake-arctic-embed** | — | petit | Snowflake, RAG |
| **qwen3-embedding** | — | variable | Qwen, embeddings |
| **all-minilm** | 384 | très petit | Sentence similarity |

---

## Vision (texte + image)

| Modèle (tag) | Paramètres | Espace disque | Contexte | Spécificité |
|--------------|------------|---------------|----------|-------------|
| **llava** / **llava-llama3** | 7B–13B | 4–8 GB | 4K–8K | LLaVA, dialogue image |
| **llama3.2-vision** | 11B, 90B | 8 GB, 50+ GB | 128K | Meta, vision officielle |
| **gemma3:4b** / **:12b** / **:27b** | 4B–27B | 3–17 GB | 128K | Google, multimodal intégré |
| **qwen2.5vl** / **qwen3-vl** | 7B+ | variable | long | Qwen, vision + langage |
| **minicpm-v** | — | petit | — | Vision compact |
| **moondream** | — | très petit | — | Vision léger |
| **bakllava** | — | variable | — | Vision, dérivé |

---

## Ordre de grandeur VRAM (inférence)

| Taille modèle | VRAM conseillée (ordre de grandeur) |
|---------------|-------------------------------------|
| 1B–3B        | 4–6 GB                              |
| 7B–8B        | 8 GB                                 |
| 12B–14B      | 12–16 GB                             |
| 22B–27B      | 16–24 GB                             |
| 32B–34B      | 24–32 GB                             |
| 70B          | 40–48 GB                             |
| 405B+        | 200+ GB (multi-GPU / quant fort)     |

Les valeurs dépendent de la quantification (Q4, Q5, Q6, etc.) : plus la quant est forte, moins ça prend de VRAM et un peu plus de disque.

---

*D’après la doc Ollama, la librairie officielle et les fiches modèles (ollama.com/library). Pour une liste à jour : `ollama list` après installation, ou consulter [ollama.com/library](https://ollama.com/library).*
