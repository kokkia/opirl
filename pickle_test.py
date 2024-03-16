import pickle
import joblib
import numpy as np

dump = False
load = True

EXPERT_PATH = "experts/sac/halfcheetah/expert/expert_00200000_epi_00_return_07186.7072.pkl"

if dump:
    array = np.array([1, 2, 3])
    print(array)

    with open("test.pickle", "wb") as f:
        pickle.dump(array, f)

if load:
    # with open("test.pickle", "rb") as f:
    # with open(EXPERT_PATH, "rb") as f:
    #     array_load = pickle.load(f)
    array_load = joblib.load(EXPERT_PATH)
    print(array_load, array_load["obs"].shape[0])
