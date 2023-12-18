#include "pyinterface.h"

#define CHECK_GIL if (PyGILState_Check()) printf("GIL is currently held by this thread.\n"); else printf("GIL is not held by this thread.\n");
#define AQUIRE_GIL PyGILState_STATE gState = PyGILState_Ensure();
#define RELEASE_GIL PyGILState_Release(gState); if (PyGILState_Check()) PyEval_SaveThread();

void print_bytes(bytes_t cBytes) {
  printf("C: ");
  for (size_t i = 0; i < 10; i++) {
    printf("%02X ", (unsigned char)cBytes.data[i]);
  }
  printf("... ");

  for (size_t i = cBytes.size - 10; i < cBytes.size; i++) {
    printf("%02X ", (unsigned char)cBytes.data[i]);
  }
  printf("\n");
}

PyObject* pyModule;

PyObject* pyUpdateDataset;
PyObject* pyUpdateDataloader;
PyObject* pyGetNumBatches;
PyObject* pyGetCleanBatch;

int initPython() {
  printf("C: initPython - start\n");
  Py_Initialize();

  PyObject* sys = PyImport_ImportModule("sys");
  PyObject* path = PyObject_GetAttrString(sys, "path");
  PyList_Append(path, PyUnicode_DecodeFSDefault("."));

  PyObject* ModuleString = PyUnicode_DecodeFSDefault((char*)"dataloader");
 
  if (!ModuleString) 
    PyErr_Print();

  PyObject* pyModule = PyImport_Import(ModuleString);
  
  if (!pyModule) 
    PyErr_Print();

  if (pyModule != NULL)
    Py_DECREF(pyModule);
  else
    PyErr_Print();

  pyUpdateDataset = PyObject_GetAttrString(pyModule, "update_dataset");
  pyUpdateDataloader = PyObject_GetAttrString(pyModule, "update_dataloader");
  pyGetNumBatches = PyObject_GetAttrString(pyModule, "get_num_batches");
  pyGetCleanBatch = PyObject_GetAttrString(pyModule, "get_clean_batch");
  printf("C: initPython - end\n");
  return 0;
}

int finalizePython() {
  printf("C: finalizePython - start\n");
  Py_DECREF(pySetDevice);
  Py_DECREF(pyPerturb);
  Py_DECREF(pyUpdateAttack);
  Py_DECREF(pyUpdateModel);
  Py_DECREF(pyModule);
  Py_Finalize();  // This causes a segfault. Further debugging is needed!
  printf("C: finalizePython - end\n");
  return 0;
}

int updateDataset(bytes_t inputBytes) {
  AQUIRE_GIL

  PyObject* pyBytes = PyMemoryView_FromMemory(inputBytes.data, inputBytes.size, PyBUF_READ);
  PyObject* pyArgs = PyTuple_Pack(1, pyBytes);
  PyObject* pyResult = PyObject_CallObject(pyUpdateDataset, pyArgs);

  Py_DECREF(pyBytes);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  RELEASE_GIL
  return 0;
}

int updateDataloader(bytes_t inputBytes) {
  AQUIRE_GIL

  PyObject* pyBytes = PyMemoryView_FromMemory(inputBytes.data, inputBytes.size, PyBUF_READ);
  PyObject* pyArgs = PyTuple_Pack(1, pyBytes);
  PyObject* pyResult = PyObject_CallObject(pyUpdateDataloader, pyArgs);

  Py_DECREF(pyBytes);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  RELEASE_GIL
  return 0;
}

size_t getNumBatches() {
  AQUIRE_GIL

  PyObject* pyArgs = PyTuple_New(0);
  PyObject* pyResult = PyObject_CallObject(pyGetNumBatches, pyArgs);

  size_t result = PyLong_AsSize_t(pyResult);

  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  RELEASE_GIL
  return result;
}

bytes_t getCleanBatch() {
  AQUIRE_GIL

  PyObject* pyArgs = PyTuple_New(0);
  PyObject* pyResult = PyObject_CallObject(pyGetCleanBatch, pyArgs);

  char* pyInternalBytes = PyBytes_AsString(pyResult);
  size_t numBytes = PyBytes_Size(pyResult);

  bytes_t outputBytes = (bytes_t){(int8_t*)malloc(numBytes), numBytes};
  // Copy the data to not refer to the internal Python memory.
  memcpy(pyInternalBytes, outputBytes.data, outputBytes.size);

  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  RELEASE_GIL
  return outputBytes;
}
