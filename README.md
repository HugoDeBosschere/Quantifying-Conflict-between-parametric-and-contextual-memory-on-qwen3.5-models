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
ou container ajouter au préalable les nouveaux modules .py et ajouter l'import correpsondant dans le fichier qui sera executé




~/test

-- WrapRotatedNumpy -- 
un wrapper qui modifie les fonctions de Numpy en changeant l'ordre des arguments de façon cyclique
à noter qu'il est callable comment un module à par entière.



-- dataRotated.jsonl -- 
un test de format de questions à poser à notre LLM. à améliorer car trop direct comme usage.


-- RotatedNumpy -- 
autre test de modif de numpy mais moins abouti






Sources improtantes -- Etat de l'art : 

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








- Humaneval : un classique d'évalusation de code, on peut s'en inspirer pour leurs formats de questions
  https://github.com/openai/human-eval/blob/master/data/example_samples.jsonl


- Big code evaluation harness : IMPORTANT - en gros c'est un papier qui présente un framework hyper cohérent avec ce qu'on cherche
  Ce repo permet de définir des tâches qui sont des :     """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
  en gros on fournit un dataset de questions réponses, test unitaire etc. et ensuite LE PLUS IMPORTANT, il montre comment 
  on peut à partir de la réponse du LLM : 1) récupérer le code proprement et le clean pour être executable et ensuite 
  2) pouvoir executer le code sur une instance type container afin de vérifier concrêtement si le code est fonctionnel ou non
  ---> toute la partie execution de code dans un container est le point clé de comment on pourrait faire pour vérif nos réponses
  https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/base.py

- 
