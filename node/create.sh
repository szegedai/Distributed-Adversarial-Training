#!/bin/bash 

docker run --gpus all -it -d --name node --mount type=bind,source="$(pwd)",target=/node node
