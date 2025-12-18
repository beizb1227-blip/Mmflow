import os
import pickle

file_path = "data/sdd/original/sdd_test.pkl"

if os.path.exists(file_path):
    try:
        with open(file_path, "rb") as file:
            pickle.load(file)
    except (pickle.UnpicklingError, EOFError, FileNotFoundError):
        print("Invalid pickle file. The file may be corrupted.")
    else:
        print("Pickle file is valid.")
else:
    print("Pickle file does not exist.")
