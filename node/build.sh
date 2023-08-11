#!/bin/bash

gcc -I/usr/include/python3.10 -fPIC -shared -o libpy_test.so py_test.c -L/usr/lib/python3.10/config-3.10-aarch64-linux-gnu -lpython3.10

go build -ldflags="-extldflags '-Wl,-rpath,.' -L ." main.go

