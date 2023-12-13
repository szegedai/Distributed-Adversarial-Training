#!/bin/bash

PY_VERSION=$(basename /usr/include/python3.*)
gcc -I/usr/include/$PY_VERSION -fPIC -shared -o libpy_wrapper.so py_wrapper.c -L$(realpath /usr/lib/$PY_VERSION/config-*) -l$PY_VERSION

export CGO_CFLAGS="-I/usr/include/$PY_VERSION"
export CGO_LDFLAGS="-L. -lpy_wrapper"
go build -ldflags="-extldflags '-Wl,-rpath,.' -L ." node.go

