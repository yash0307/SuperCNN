#!/bin/bash

export CUDA_HOME=/usr/local/cuda-8.0/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
PATH=${CUDA_HOME}/bin:${PATH} 
export PATH
export CUDA_VISIBLE_DEVICES=3
