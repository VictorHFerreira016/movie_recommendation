# joblib is a library used to 
import joblib

# save and load Python objects efficiently.
# It is particularly useful for saving machine learning models and large datasets.
def save_artifact(obj, path):
    # The function dump saves the object to the specified path.
    joblib.dump(obj, path)

def load_artifact(path):
    # The function load loads the object from the specifie path.
    return joblib.load(path)