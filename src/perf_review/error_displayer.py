import json

cpt = 0
with open("/usr/users/sdim/sdim_25/memory_code_eval/src/perf_review/error_gathored.txt", mode = 'r') as f:
    with open("/usr/users/sdim/sdim_25/memory_code_eval/src/perf_review/results/result_try.jsonl", mode = 'r') as f_data:

        for line in f_data :
            res = json.loads(line)
            if res.get("stderr", ""):
                print(res.get("stderr", ""))
                f.write(res.get("stderr", ""))
                f.write("\n\n")
                cpt+=1
    f.write(f"Total errors : {cpt}\n")
    print(cpt)