#!/bin/bash 

docker run --gpus all -it --network host -d --name node --mount type=bind,source="$(pwd)",target=/node node
