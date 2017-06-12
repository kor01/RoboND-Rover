#! /bin/bash

target_dir=$1

mkdir ${target_dir}

mkdir ${target_dir}/config ${target_dir}/environment

mkdir ${target_dir}/perception ${target_dir}/reactor

mkdir ${target_dir}/proactor ${target_dir}/replay

cp code/config/agent_spec.cfg ${target_dir}/config

cp code/config/game_spec.cfg ${target_dir}/config

