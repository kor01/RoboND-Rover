import os

DIR_NAME = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_DIR = os.path.join(dir_name, 'experiments')

def experiment(path):
    return os.path.join(EXPERIMENT_DIR, path)
