FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y python3.10 python3.10-distutils python3.10-dev python3-pip
RUN pip install numpy torch dill

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y golang-go
