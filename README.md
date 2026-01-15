Listes d'idées/état de l'art/ avancées


But :
On veut vérifier à quel point les LLM viennent à utiliser leur mémoire paramétrique our leur mémoire factuelle
ie : mémoire contextuelle/RAG. 

idée de base:
créer une nouvelle librairie (faire des wrapper de librairie pour construire des nouvelles fonctions : noms différents,
inversion d'arguments, --> essayer de trouver plusieurs niveaux de modif) fournir cette nouvelle librairie en contexte.

Une fois fait on veut pouvoir donner des tests unitaires à notre LLM en lui donnant la nouvelle librairie (et en lui 
disant de l'utiliser). L'idée serait donc de lui demander de résoudre des petits problèmes qui utilisent des fonctions numpy
et de controller de façon fonctionnelle (en executant le code de sortie) pour savoir s'il a intégré la modif ou non.

Pour s'assurer que ce sont bien les fonctions de notre nouvelle librairie qui sont utilisées on peut dans la sandbox
ou container ajouter au préalable les nouveaux modules .py et ajouter l'import correspondant dans le fichier qui sera executé




~/test

-- WrapRotatedNumpy -- 
un wrapper qui modifie les fonctions de Numpy en changeant l'ordre des arguments de façon cyclique
à noter qu'il est callable comment un module à par entière.



-- dataRotated.jsonl -- 
un test de format de questions à poser à notre LLM. à améliorer car trop direct comme usage.


-- RotatedNumpy -- 
autre test de modif de numpy mais moins abouti




BUT intermédiaire : intégrer le fait de pouvoir rajouter plusieurs llm et plusieurs lib sur le fichier de config de telle sorte que les executions en fonction d'un contexte (description, example, tout le fichier) se fassent toutes d'un coup selon ce qu'on ajoute ou non sur le fichier de config. 

En gros rendre la pipeline la plus générique possible pour que plus tard seul la partie écriture de fichier temporaire d'execution etc dépende uniquement de comment la data est présentée et on conserve les autres fichiers.

faire un fichier d'exec pour lancer les commandes bash de ollama etc au préalable pour une meilleure automatisation.

Faire une image docker, un dockerfile pour que les exec se fassent dessus. --> meilleure sécurité.

Voir comment on peut nous même créer notre benchmark de questions à tester. créer le bench en fonction de nos besoins. 










Sources importantes -- Etat de l'art : 

- Papier OpenAI expliquant comment ils créent des SandBox afin d'executer du code provenant de LLM de façon automatisé:
  https://arxiv.org/pdf/2107.03374


- Dataset de questions python : à creuser mais contient 220 questions ciblés sur numpy:
  https://github.com/xlang-ai/DS-1000/tree/main

data example :

{
    "prompt": problem description, 
    "reference_code": The working code for the problem, 
    "metadata": {
                "problem_id": 300, 
                "library_problem_id": 9, 
                "library": "Numpy", 
                "test_case_cnt": 2, 
                "perturbation_type": "Origin", 
                "perturbation_origin_id": 9
                }, 
    "code_context": a code that contains a generation of examples and assertions
}





- Humaneval : un classique d'évaluation de code, on peut s'en inspirer pour leurs formats de questions
  https://github.com/openai/human-eval/blob/master/data/example_samples.jsonl


- Big code evaluation harness : IMPORTANT - en gros c'est un papier qui présente un framework hyper cohérent avec ce qu'on cherche
  Ce repo permet de définir des tâches qui sont des :     
  """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
  en gros on fournit un dataset de questions réponses, test unitaire etc. et ensuite LE PLUS IMPORTANT, il montre comment 
  on peut à partir de la réponse du LLM : 1) récupérer le code proprement et le clean pour être executable et ensuite 
  2) pouvoir executer le code sur une instance type container afin de vérifier concrêtement si le code est fonctionnel ou non
  ---> toute la partie execution de code dans un container est le point clé de comment on pourrait faire pour vérif nos réponses
  https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/base.py

différente manière d'explqiuer le contexte au llm : 
                                {
                                "description": "A recursive proxy wrapper for the NumPy library that systematically intercepts function calls and performs a 'left rotation' on their positional arguments. Essentially, for any function called, the first argument is moved to the last position before the actual NumPy calculation is executed, while constants and module structures remain accessible.",
                                "explanation": "The library works by hacking `sys.modules` to replace itself with an instance of the `RotatedNumPy` class. This class uses `__getattr__` to dynamically inspect whatever you are trying to access from NumPy.\n\n1. **Constants:** If you access a variable (like `np.pi`), it passes it through unchanged.\n2. **Submodules:** If you access a submodule (like `np.linalg`), it wraps that submodule in a new `RotatedNumPy` instance so the behavior persists deeper in the library.\n3. **Functions:** If you access a function (like `np.subtract`), it wraps it in a decorator (`rotate_args_logic`). This decorator takes the arguments provided `(a, b, c)`, moves the first one to the end `(b, c, a)`, and then runs the original function.\n\n**Example:**\nStandard NumPy `subtract` takes `(x1, x2)` and calculates `x1 - x2`.\nIf you call `lib.subtract(10, 5)`:\n- The wrapper receives `(10, 5)`.\n- It rotates them to `(5, 10)`.\n- It calls real numpy: `5 - 10`.\n- Result: `-5` (instead of 5).",
                                "whole_lib": "import numpy as _real_np\nimport types\nimport sys\n\n# 1. Ton wrapper de rotation (inchangé)\ndef rotate_args_logic(func):\n    def wrapper(*args, **kwargs):\n        args_list = list(args)\n        if not args_list:\n            return func(*args, **kwargs)\n        # Rotation de 1 vers la gauche : le 1er passe à la fin\n        new_args = args_list[1:] + args_list[:1]\n        return func(*new_args, **kwargs)\n    return wrapper\n\n# 2. La classe \"Proxy\" qui intercepte tout\nclass RotatedNumPy:\n    def __init__(self, target_module):\n        self._target = target_module\n\n    def __getattr__(self, name):\n        # On récupère l'objet réel dans NumPy (ex: np.add ou np.linalg)\n        real_attr = getattr(self._target, name)\n\n        # CAS 1 : C'est un sous-module (ex: np.linalg)\n        # On retourne un nouveau Proxy pour ce sous-module\n        if isinstance(real_attr, types.ModuleType):\n            return RotatedNumPy(real_attr)\n\n        # CAS 2 : C'est une fonction ou une ufunc (ex: np.add, np.mean)\n        # On exclut les \"types\" (comme np.int32, np.float64, np.array) car ce sont des classes\n        if callable(real_attr) and not isinstance(real_attr, type):\n            return rotate_args_logic(real_attr)\n\n        # CAS 3 : C'est une constante (ex: np.pi, np.nan)\n        # On retourne la valeur telle quelle\n        return real_attr\n    \n    # Permet à dir(mon_numpy) de montrer les mêmes choses que dir(numpy)\n    def __dir__(self):\n        return dir(self._target)\n\n    # Permet d'afficher l'objet proprement\n    def __repr__(self):\n        return f\"<RotatedNumPy Proxy sur {self._target.__name__}>\"\n\n# 3. L'astuce finale\n# On remplace le module actuel par une instance de notre classe.\n# Cela permet d'utiliser \"import mon_numpy\" comme si c'était un vrai module.\nsys.modules[__name__] = RotatedNumPy(_real_np)"
                              }