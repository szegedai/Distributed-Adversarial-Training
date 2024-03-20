#!/bin/bash

PY_VERSION=$(basename /usr/include/python3.*)
gcc -I/usr/include/$PY_VERSION -fPIC -shared -o libpyinterface.so pyinterface.c -L$(realpath /usr/lib/$PY_VERSION/config-*) -l$PY_VERSION

export CGO_CFLAGS="-I/usr/include/$PY_VERSION"
export CGO_LDFLAGS="-L. -lpyinterface"
/usr/local/go/bin/go build -ldflags="-extldflags '-Wl,-rpath,.' -L ." node.go

