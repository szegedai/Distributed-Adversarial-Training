package main

/*
#include "pyinterface.h"
*/
import "C"
import (
	"bytes"
	"encoding/binary"
  "encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/signal"
	"reflect"
	"sync"
	"syscall"
	"unsafe"
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
  /*sliceHeader := reflect.SliceHeader{
    Data: uintptr(unsafe.Pointer(b.data)),
    Len: (int)(b.size),
    Cap: (int)(b.size),
  }
  return *(*[]byte)(unsafe.Pointer(&sliceHeader))*/
  bytes := C.GoBytes(unsafe.Pointer(b.data), (C.int)(b.size))
  C.free(unsafe.Pointer(b.data))
  return bytes
}

type Node struct {
  Host string
  Device string
  BufferSize uint16
  session http.Client
  mainWG sync.WaitGroup
  running bool
  attackID uint64
  modelID uint64
  modelStateID uint64
}

func (self *Node) Run() {
  // TODO: Check if a valid host name and device was given!

  log.Println("Starting node with config: { Host:", self.Host, "Device:", self.Device, "BufferSize:", self.BufferSize, "}")

  self.running = true
  c := make(chan os.Signal)
  signal.Notify(c, os.Interrupt, syscall.SIGTERM)
  go func() {
    <-c
    log.Println("\nThe program will terminate after finishing the current batch...")
    self.running = false
    <-c
    os.Exit(0)
  }()

  C.initPython()
  defer C.finalizePython()

  self.session = http.Client{}

  self.mainWG = sync.WaitGroup{}

  self.setDevice()

  log.Println("Ready and running")

  self.mainWG.Add(1)
  go func() {
    self.attackID, self.modelID, self.modelStateID = self.getIDs()
  }()

  self.mainWG.Add(1)
  go func() {
    defer self.mainWG.Done()
    C.updateModel(GB2CB(self.getData("/model")))
    C.updateAttack(GB2CB(self.getData("/attack")))
    C.pushModelState(GB2CB(self.getData("/model_state")))
  }()

  for i := (uint16)(0); i < self.BufferSize; i++ {
    self.mainWG.Add(1)
    go func() {
      defer self.mainWG.Done()
      C.pushBatch(GB2CB(self.getData("/clean_batch")))
    }()
  }

  self.mainWG.Wait()

  for self.running {
    self.mainWG.Add(1)
    go func() {
      defer self.mainWG.Done()

      self.mainWG.Add(1)
      latestAttackID, latestModelID, latestModelStateID := self.getIDs()

      if latestModelStateID != self.modelStateID {
        self.modelStateID = latestModelStateID

        self.mainWG.Add(1)
        go func() {
          defer self.mainWG.Done()
          C.pushModelState(GB2CB(self.getData("/model_state")))
        }()
      }

      if latestAttackID != self.attackID {
        self.attackID = latestAttackID

        self.mainWG.Add(1)
        go func() {
          defer self.mainWG.Done()
          C.updateAttack(GB2CB(self.getData("/attack")))
        }()
      }
      
      if latestModelID != self.modelID {
        self.modelID = latestModelID

        self.mainWG.Add(1)
        go func() {
          defer self.mainWG.Done()
          C.updateModel(GB2CB(self.getData("/model")))
        }()
      }

    }()

    batchBytes := CB2GB(C.popBatch())

    self.mainWG.Add(2)
    go func() {
      defer self.mainWG.Done()
      self.postData("/adv_batch", batchBytes, struct{ ModelStateID uint64 }{self.modelStateID})
    }()
    go func() {
      defer self.mainWG.Done()
      C.pushBatch(GB2CB(self.getData("/clean_batch")))
    }()

    self.mainWG.Wait()
  }
}

func (self *Node) setDevice() {
  cDevice := C.CString(self.Device)
  defer C.free(unsafe.Pointer(cDevice))
  C.setDevice(cDevice)
}

func (self *Node) getData(resource string) []byte {
  resp, err := self.session.Get(self.Host + resource)
  if err != nil {
    log.Fatal(err)
  }
  defer resp.Body.Close()

  body, err := ioutil.ReadAll(resp.Body)
  if err != nil {
    log.Fatal(err)
  }

  return body
}

func (self *Node) postData(resource string, data []byte, extraData any) {
  req, err := http.NewRequest("POST", self.Host + resource, bytes.NewBuffer(data))
  if err != nil {
    log.Fatal(err)
  }

  req.Header.Set("Content-Type", "application/octet-stream")

  if extraData != nil {
    jsonString, err := json.Marshal(extraData)
    if err != nil {
      log.Fatal(err)
    }
    req.Header.Set("X-Extra-Data", string(jsonString))
  }

  resp, err := self.session.Do(req)
  if err != nil {
    log.Fatal(err)
  }
  /*resp, err := self.session.Post(self.Host + resource, "application/octet-stream", bytes.NewBuffer(data))
  if err != nil {
    log.Fatal(err)
  }*/
  defer resp.Body.Close()
}

func (self *Node) getIDs() (uint64, uint64, uint64) {
  defer self.mainWG.Done()

  data := self.getData("/ids")
  return binary.BigEndian.Uint64(data[:8]), binary.BigEndian.Uint64(data[8:16]), binary.BigEndian.Uint64(data[16:])
}


func main() {
  host := flag.String("H", "http://127.0.0.1:8080", "The exact host where the server is running")
  device := flag.String("D", "cpu", "The device the is used by PyTorch for the perturbation process")
  bufferSize := flag.Uint("B", 2, "The amount of batches preloaded by the node")

  flag.Parse()

  n := Node{Host: *host, Device: *device, BufferSize: (uint16)(*bufferSize)}
  n.Run()
}
