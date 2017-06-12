#! /bin/bash

set -x

project_root=$(dirname $(pwd))

export PYTHONPATH=${PYTHONPATH}:${project_root}/code

export PYTHONPATH=${PYTHONPATH}:${project_root}/notebook

nohup jupyter-notebook --port=8889 > /dev/null &
