import os
from tree_of_thoughts.treeofthoughts import OpenAILanguageModel, OptimizedOpenAILanguageModel, TreeofThoughts, OptimizedTreeofThoughts

api_key = os.getenv("OPENAI_API_KEY")
api_base = ""  # leave it blank if you simply use default openai api url

use_v2 = False

if not use_v2:
    # v1
    model = OpenAILanguageModel(api_key=api_key, api_base=api_base)
else:
    # v2 parallel execution, caching, adaptive temperature
    model = OptimizedOpenAILanguageModel(api_key=api_key, api_base=api_base)

# choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

if not use_v2:
    # create an instance of the tree of thoughts class v1
    tree_of_thoughts = TreeofThoughts(model, search_algorithm)
else:
    # or v2
    tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"

# Parameters
k = 5
T = 3
b = 5
vth = 0.5
timeout = 10
confidence = 1.0  # model is confident on performance
max_iterations = 40  # tree branch nodes
convergence_threshold = 0.01
convergence_count = 5

solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, timeout, confidence_threshold=confidence, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)

print(f'solution: {solution}')
