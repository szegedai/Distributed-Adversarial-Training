#include "py_wrapper.h"

PyObject* pyModule;

PyObject* pySetDevice;
PyObject* pyPerturb;
PyObject* pyUpdateAttack;
PyObject* pyUpdateModel;

int initPython() {
  Py_Initialize();

  PyObject* pyName = PyUnicode_DecodeFSDefault("perturber");
  PyObject* pyModule = PyImport_Import(pyName);
  Py_DECREF(pyName);

  pySetDevice = PyObject_GetAttrString(pyModule, "set_device");
  pyPerturb = PyObject_GetAttrString(pyModule, "perturb");
  pyUpdateAttack = PyObject_GetAttrString(pyModule, "update_attack");
  pyUpdateModel = PyObject_GetAttrString(pyModule, "update_model");
  return 0;
}

int finalizePython() {
  Py_DECREF(pySetDevice);
  Py_DECREF(pyPerturb);
  Py_DECREF(pyUpdateAttack);
  Py_DECREF(pyUpdateModel);
  Py_DECREF(pyModule);
  Py_Finalize();
  return 0;
}

int setDevice(char* newDevice) {
  PyGILState_STATE gState = PyGILState_Ensure();

  PyObject* pyString = PyUnicode_FromString(newDevice);
  PyObject* pyArgs = PyTuple_Pack(1, pyString);
  PyObject* pyResult = PyObject_CallObject(pySetDevice, pyArgs);

  Py_DECREF(pyString);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  PyGILState_Release(gState);
  return 0;
}

bytes_t perturb(bytes_t inputBytes) {
  PyGILState_STATE gState = PyGILState_Ensure();

  PyObject* pyBytes = PyBytes_FromStringAndSize(inputBytes.data, inputBytes.size);
  PyObject* pyArgs = PyTuple_Pack(1, pyBytes);
  PyObject* pyResult = PyObject_CallObject(pyPerturb, pyArgs);

  bytes_t outputBytes = {(int8_t*)PyBytes_AsString(pyResult), (size_t)PyBytes_Size(pyResult)};

  Py_DECREF(pyBytes);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  PyGILState_Release(gState);
  return outputBytes;
}

int updateAttack(bytes_t inputBytes) {
  PyGILState_STATE gState = PyGILState_Ensure();

  PyObject* pyBytes = PyBytes_FromStringAndSize(inputBytes.data, inputBytes.size);
  PyObject* pyArgs = PyTuple_Pack(1, pyBytes);
  PyObject* pyResult = PyObject_CallObject(pyUpdateAttack, pyArgs);

  Py_DECREF(pyBytes);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  PyGILState_Release(gState);
  return 0;
}

int updateModel(bytes_t inputBytes) {
  PyGILState_STATE gState = PyGILState_Ensure();

  PyObject* pyBytes = PyBytes_FromStringAndSize(inputBytes.data, inputBytes.size);
  PyObject* pyArgs = PyTuple_Pack(1, pyBytes);
  PyObject* pyResult = PyObject_CallObject(pyUpdateModel, pyArgs);

  Py_DECREF(pyBytes);
  Py_DECREF(pyArgs);
  Py_DECREF(pyResult);

  PyGILState_Release(gState);
  return 0;
}

int main() {
  initPython();
  printf("asd\n");
  finalizePython();
  return 0;
}
