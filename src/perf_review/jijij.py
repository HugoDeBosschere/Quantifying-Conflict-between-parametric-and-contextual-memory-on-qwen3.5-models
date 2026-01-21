import json

cpt = 0
with open("/usr/users/sdim/sdim_25/memory_code_eval/src/perf_review/results/result_try.jsonl", mode = 'r') as f:

    for line in f :
        res = json.loads(line)
        if res.get("stderr", ""):
            print(res.get("stderr", ""))
            cpt+=1
print(cpt)