FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch==2.1.2 torchvision==0.16.2 cloudpickle==3.0.0

RUN apt-get install -y curl
RUN curl -O https://dl.google.com/go/go1.21.5.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
ENV PATH="${PATH}:/usr/local/go/bin"
