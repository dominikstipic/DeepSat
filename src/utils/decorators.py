
import functools
import pdb


def analyzer(fun):
    @functools.wraps(fun)
    def wrapper(*argv, **kwargs):
        print(f"Poziv Funkcije {fun.__name__}:\n", 
              f"Pozicijski argumenti: {argv}\n",
              f"Imenovani argumenti: {kwargs}")
        rv = fun(*argv, **kwargs)
        print(f"Povratna vrijednost: {rv}")
        return rv
    return wrapper

def timer(fun):
    import time
    import numpy as np
    @functools.wraps(fun)
    def wrapper(*argv, **kwargs):
        n = 10000
        xs = np.zeros(n)
        for i in range(n):
            start = time.perf_counter()
            fun(*argv, **kwargs)
            end = time.perf_counter()
            xs[i] = end-start
        print(f"total = {xs.sum()}\n",
              f"mean = {xs.mean()}\n",
              f"std = {xs.std()}\n", 
              f"n = {n}")
    return wrapper


def safe_interruption(fun):
    @functools.wraps(fun)
    def wrapper(*argv, **kwargs):
        try:
            return fun(*argv, **kwargs)
        except Exception:
            print("Execution stoped")
    return wrapper


def debug(fun):
    @functools.wraps(fun)
    def wrapper(*argv, **kwargs):
        pdb.set_trace()
        fun(*argv, **kwargs)
    return wrapper

def signal(fun):
    @functools.wraps(fun)
    def wrapper(*argv, **kwargs):
        print("Signal")
        fun(*argv, **kwargs)
    return wrapper

if __name__ == "__main__":
    @safe_interruption
    def fun():
        while True:
            pass
    
    #fun()