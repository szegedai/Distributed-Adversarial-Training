#ifndef PY_WRAPPER_H
#define PY_WRAPPER_H

#include <Python.h>
#include <stdlib.h>

typedef struct {
  int8_t* data;
  size_t size;
} bytes_t;

extern PyObject* pyModule;

extern PyObject* pySetDevice;
extern PyObject* pyPerturb;
extern PyObject* pyUpdateAttack;
extern PyObject* pyUpdateModel;

int initPython();

int finalizePython();

int setDevice(char* newDevice);

int perturb(bytes_t inputBytes);

int updateAttack(bytes_t inputBytes);

int updateModel(bytes_t inputBytes);

#endif
