#!/bin/bash

gcc -I/usr/include/python3.10 -fPIC -shared -o libpy_wrapper.so py_wrapper.c -L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -lpython3.10

go build -ldflags="-extldflags '-Wl,-rpath,.' -L ." node.go

