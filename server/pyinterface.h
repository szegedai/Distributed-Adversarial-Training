#ifndef PYINTERFACE_H
#define PYINTERFACE_H

#include <Python.h>
#include <stdlib.h>

typedef struct {
  int8_t* data;
  size_t size;
} bytes_t;

extern PyObject* pyModule;

extern PyObject* pyUpdateData;
extern PyObject* pyGetNumBatches;
extern PyObject* pyGetCleanBatch;

int initPython();

int finalizePython();

int updateData(bytes_t inputBytes);

size_t getNumBatches();

bytes_t getCleanBatch();

#endif
