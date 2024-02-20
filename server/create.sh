#!/bin/bash 

docker run --gpus all -it --network host -d --name server --mount type=bind,source="$(pwd)",target=/server --mount type=bind,source=/home/developer/imagenet,target=/imagenet --shm-size 20G server
