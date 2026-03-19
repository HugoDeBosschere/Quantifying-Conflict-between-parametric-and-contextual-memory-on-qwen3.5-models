
import inspect
import numpy 
import re

def no_kwargs(func_name):
    ord_dict = inspect.signature(func_name).parameters
    print(ord_dict)
    pos_args = []
    for k,v in ord_dict.items():
        if v.kind == inspect.Parameter.POSITIONAL_ONLY or v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            pos_args.append(v)

    return pos_args


def add_underscore(s):
    new_s = re.sub(r"np\.(.*?) ",r"np.\1_ ", s) 
    new_s = re.sub(r"np\.(.*?)\Z",r"np.\1_", s) 
    return new_s

if __name__ == "__main__" :
    text = "jdsfjs np.nonjour() np.djhfej np.coucou"
    print(add_underscore(text))