FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch==2.0.1 dill==0.3.7 torchvision

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y golang-go
