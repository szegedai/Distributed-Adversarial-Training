package main

/*
#cgo CFLAGS: -I/usr/include/python3.10
#cgo LDFLAGS: -L. -lpy_wrapper
#include "py_wrapper.h"
*/
import "C"
import (
  "encoding/binary"
  "fmt"
  "io/ioutil"
  "net/http"
  "unsafe"
  "reflect"
  "sync"
  "bytes"
  "os"
  "os/signal"
  "syscall"
)

func printBytes(data []byte) {
  fmt.Print("Go: ")
  for i := 0; i < 10; i++ {
    fmt.Printf("%02X ", data[i])
  }
  fmt.Print("... ")

  start := len(data) - 10
  for i := start; i < len(data); i++ {
    fmt.Printf("%02X ", data[i])
  }
  fmt.Println()
}

func GB2CB(b []byte) C.bytes_t {
  data := (*C.int8_t)(unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&b)).Data))
  size := (C.size_t)(len(b))
  return C.bytes_t{data, size}
}

func CB2GB(b C.bytes_t) []byte {
  sliceHeader := reflect.SliceHeader{
    Data: uintptr(unsafe.Pointer(b.data)),
    Len: (int)(b.size),
    Cap: (int)(b.size),
  }
  return *(*[]byte)(unsafe.Pointer(&sliceHeader))
}

type Node struct {
  Host string
  Device string
  BufferSize uint16
  cleanBatchBuffer chan []byte
  advBatchBuffer chan []byte
  session http.Client
  mainWG sync.WaitGroup
  running bool
  attackID uint64
  modelID uint64
}

func (self *Node) Run() {
  // TODO: Check if a valid host name and device was given!

  self.running = true
  c := make(chan os.Signal)
  signal.Notify(c, os.Interrupt, syscall.SIGTERM)
  go func() {
    <-c
    fmt.Println("\nThe program will terminate after finishing the current batch...")
    self.running = false
    <-c
    os.Exit(0)
  }()

  self.cleanBatchBuffer = make(chan []byte, self.BufferSize)
  self.advBatchBuffer = make(chan []byte, self.BufferSize)

  C.initPython()
  defer C.finalizePython()

  self.session = http.Client{}

  self.mainWG = sync.WaitGroup{}

  self.setDevice()

  self.mainWG.Add(3 + (int)(self.BufferSize))
  go func() {
    self.attackID, self.modelID = self.getIDs()
  }()
  go func() {
    self.updateAttack()
    self.updateModel()
  }()
  for i := (uint16)(0); i < self.BufferSize; i++ {
    go self.getCleanBatch()
  }
  self.mainWG.Wait()
  
  for self.running {
    self.mainWG.Add(3)
    go func() {
      fmt.Println("Go: _func_0 - start")
      defer self.mainWG.Done()
      self.advBatchBuffer <- CB2GB(C.perturb(GB2CB(<-self.cleanBatchBuffer)))
      go self.postAdvBatch()
      go self.getCleanBatch()
      fmt.Println("Go: _func_0 - end")
    }()
    self.mainWG.Add(1)
    go func() {
      fmt.Println("Go: _func_1 - start")
      defer self.mainWG.Done()
      latestAttackID, latestModelID := self.getIDs()
      if latestAttackID != self.attackID {
        self.mainWG.Add(1)
        self.attackID = latestAttackID
        go self.updateAttack()
      }
      if latestModelID != self.modelID {
	self.mainWG.Add(1)
        self.modelID = latestModelID
        go self.updateModel()
      }
      fmt.Println("Go: _func_1 - end")
    }()
    self.mainWG.Wait()
  }
}

func (self *Node) setDevice() {
  fmt.Println("Go: setDevice - start")
  cDevice := C.CString(self.Device)
  defer C.free(unsafe.Pointer(cDevice))
  C.setDevice(cDevice)
  fmt.Println("Go: setDevice - end")
}

func (self *Node) getData(resource string) []byte {
  resp, err := self.session.Get(self.Host + resource)
  if err != nil {
    panic(fmt.Sprintln("Error during GET request:", err))
  }
  defer resp.Body.Close()

  body, err := ioutil.ReadAll(resp.Body)
  if err != nil {
    panic(fmt.Sprintln("Error reading response body:", err))
  }

  return body
}

func (self *Node) postData(resource string, data []byte) {
  resp, err := self.session.Post(self.Host + resource, "application/octet-stream", bytes.NewBuffer(data))
  if err != nil {
    panic(fmt.Sprintln("Error during POST request:", err))
  }
  defer resp.Body.Close()
}

func (self *Node) getCleanBatch() {
  fmt.Println("Go: getCleanBatch - start")
  defer self.mainWG.Done()
  self.cleanBatchBuffer <- self.getData("/clean_batch")
  fmt.Println("Go: getCleanBatch - end")
}

func (self *Node) postAdvBatch() {
  fmt.Println("Go: postAdvBatch - start")
  defer self.mainWG.Done()
  self.postData("/adv_batch", <-self.advBatchBuffer)
  fmt.Println("Go: postAdvBatch - end")
}

func (self *Node) getIDs() (uint64, uint64) {
  fmt.Println("Go: getIDs - start")
  defer self.mainWG.Done()
  data := self.getData("/ids")
  fmt.Println("Go: getIDs - end")
  return binary.BigEndian.Uint64(data[:8]), binary.BigEndian.Uint64(data[8:])
}

func (self *Node) updateAttack() {
  fmt.Println("Go: updateAttack - start")
  defer self.mainWG.Done()
  data := self.getData("/attack")
  printBytes(data)
  C.updateAttack(GB2CB(data))
  fmt.Println("Go: updateAttack - end")
}

func (self *Node) updateModel() {
  fmt.Println("Go: updateModel - start")
  defer self.mainWG.Done()
  data := self.getData("/model")
  printBytes(data)
  C.updateModel(GB2CB(data))
  fmt.Println("Go: updateModel - end")
}


func main() {
  n := Node{Host: "http://127.0.0.1:8080", Device: "cpu", BufferSize: (uint16)(5)}
  n.Run()
}
