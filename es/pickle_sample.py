import pickle
import os

dict1 = {'hello' : 1, 'brother' : 2}

print(f"{os.getcwd()}/pickles/sample.pickle")
with open(f"{os.getcwd()}/pickles/sample.pickle", 'wb') as f:
    pickle.dump(dict1, f)

with open(f"{os.getcwd()}/pickles/sample.pickle", 'rb') as f:
    print(pickle.load(f))