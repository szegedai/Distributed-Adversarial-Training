FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y python3.10 python3.10-distutils python3.10-dev python3-pip
RUN pip install numpy torch dill

RUN apt-get install -y curl
RUN curl -OL https://golang.org/dl/go1.20.7.linux-arm64.tar.gz
RUN tar -C /usr/local -xzf go1.20.7.linux-arm64.tar.gz
RUN rm go1.20.7.linux-arm64.tar.gz
ENV PATH="${PATH}:/usr/local/go/bin"

COPY . /app

WORKDIR /app
RUN chmod +x build.sh
#RUN ./build.sh

#CMD ["./main"]

CMD ./build.sh && ./node
