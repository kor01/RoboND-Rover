import os
from rover_spec import PROJECT_DIR


EXPERIMENT_DIR = os.path.join(PROJECT_DIR, 'experiments')


def experiment(path):
    return os.path.join(EXPERIMENT_DIR, path)
