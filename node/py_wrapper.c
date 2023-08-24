#include "py_wrapper.h"

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

PyObject* pySetDevice;
PyObject* pyPerturb;
PyObject* pyUpdateAttack;
PyObject* pyUpdateModel;

int initPython() {
  printf("C: initPython - start\n");
  Py_Initialize();

  PyObject* sys = PyImport_ImportModule("sys");
  PyObject* path = PyObject_GetAttrString(sys, "path");
  PyList_Append(path, PyUnicode_DecodeFSDefault("."));

  PyObject* ModuleString = PyUnicode_DecodeFSDefault((char*)"perturber");
 
  if (!ModuleString) 
    PyErr_Print();

  PyObject* pyModule = PyImport_Import(ModuleString);
  
  if (!pyModule) 
    PyErr_Print();

  if (pyModule != NULL)
    Py_DECREF(pyModule);
  else
    PyErr_Print();

  pySetDevice = PyObject_GetAttrString(pyModule, "set_device");
  pyPerturb = PyObject_GetAttrString(pyModule, "perturb");
  pyUpdateAttack = PyObject_GetAttrString(pyModule, "update_attack");
  pyUpdateModel = PyObject_GetAttrString(pyModule, "update_model");
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

int setDevice(char* newDevice) {
  printf("C: setDevice - start\n");
  AQUIRE_GIL

  PyObject* pyString = PyUnicode_FromString(newDevice);
  PyObject* pyArgs = PyTuple_Pack(1, pyString);
  PyObject* pyResult = PyObject_CallObject(pySetDevice, pyArgs);

  Py_DECREF(pyString);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  RELEASE_GIL
  printf("C: setDevice - end\n");
  return 0;
}

bytes_t perturb(bytes_t inputBytes) {
  printf("C: perturb - start\n");
  AQUIRE_GIL

  PyObject* pyBytes = PyMemoryView_FromMemory(inputBytes.data, inputBytes.size, PyBUF_READ);
  PyObject* pyArgs = PyTuple_Pack(1, pyBytes);
  PyObject* pyResult = PyObject_CallObject(pyPerturb, pyArgs);

  bytes_t outputBytes = {(int8_t*)PyBytes_AsString(pyResult), (size_t)PyBytes_Size(pyResult)};

  Py_DECREF(pyBytes);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  RELEASE_GIL
  printf("C: perturb - end\n");
  return outputBytes;
}

int updateAttack(bytes_t inputBytes) {
  printf("C: updateAttack - start\n");
  AQUIRE_GIL

  PyObject* pyBytes = PyMemoryView_FromMemory(inputBytes.data, inputBytes.size, PyBUF_READ);
  print_bytes(inputBytes);
  PyObject* pyArgs = PyTuple_Pack(1, pyBytes);
  PyObject* pyResult = PyObject_CallObject(pyUpdateAttack, pyArgs);

  Py_DECREF(pyBytes);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  RELEASE_GIL
  printf("C: updateAttack - end\n");
  return 0;
}

int updateModel(bytes_t inputBytes) {
  printf("C: updateModel - start\n");
  AQUIRE_GIL

  PyObject* pyBytes = PyMemoryView_FromMemory(inputBytes.data, inputBytes.size, PyBUF_READ);
  print_bytes(inputBytes);
  PyObject* pyArgs = PyTuple_Pack(1, pyBytes);
  PyObject* pyResult = PyObject_CallObject(pyUpdateModel, pyArgs);

  Py_DECREF(pyBytes);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  RELEASE_GIL
  printf("C: updateModel - end\n");
  return 0;
}
