import json
import os
import glob
import pickle


if __name__ == "__main__":
    pickle_file_name = f"{os.getcwd()}/pickles/raw/*"

    files = glob.glob(pickle_file_name)
    for file in files:
        basename_without_ext = os.path.splitext(os.path.basename(file))[0]

        with open(file, "rb") as f1:
            docs = pickle.load(f1)
            with open(f"{os.getcwd()}/jsons/{basename_without_ext}.json", "w") as f2:
                json.dump(docs, f2, ensure_ascii=False, indent=4)
