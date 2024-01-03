package main

/*
#include "pyinterface.h"
*/
import "C"
import (
	"context"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/signal"
	"reflect"
	"syscall"
	"time"
	"unsafe"
  "sync"
	. "github.com/Jcowwell/go-algorithm-club/Heap"
	. "github.com/Jcowwell/go-algorithm-club/Utils"
)

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

func dispathRequest(pattern string, getHandler func(http.ResponseWriter, *http.Request), postHandler func(http.ResponseWriter, *http.Request)) {
  http.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodGet:
      if getHandler != nil {
        w.Header().Set("Content-Type", "application/octet-stream")
        w.WriteHeader(http.StatusOK)
        getHandler(w, r)
      } else {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
      }
    case http.MethodPost:
      if postHandler != nil {
        w.WriteHeader(http.StatusOK)
        postHandler(w, r)
      } else {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
      } 
    default:
      http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
  })
}

type Batch struct {
  clean []byte
  adv []byte
}

type Server struct {
  address string

  queueSoftLimit uint64
  maxPatiente uint64

  modelData []byte
  modelID uint64
  modelMutex sync.RWMutex
  attackData []byte
  attackID uint64
  attackMutex sync.RWMutex

  freeQ chan *Batch
  workQ sync.Map
  doneQ chan *Batch
}

func InitServer(address string) *Server {
  return &Server{
    address: address, 
    queueSoftLimit: 0, maxPatiente: 0, 
    modelData: nil, modelID: 0, 
    attackData: nil, attackID: 0, 
    freeQ: nil, 
    workQ: sync.Map{}, 
    doneQ: nil,
  }
}

func (self *Server) Run() {
  stop := make(chan os.Signal)
  signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

  C.initPython()
  //defer C.finalizePython()  //Debuging required!

  dispathRequest("/attack", self.onGetAttack, self.onPostAttack)
  dispathRequest("/model", self.onGetModel, self.onPostModel)
  dispathRequest("/adv_batch", self.onGetAdvBatch, self.onPostAdvBatch)
  dispathRequest("/clean_batch", self.onGetCleanBatch, nil)
  dispathRequest("/ids", self.onGetIDs, nil)
  dispathRequest("/num_batches", self.onGetNumBatches, nil)
  dispathRequest("/dataset", nil, self.onPostDataset)
  dispathRequest("/dataloader", nil, self.onPostDataloader)

  httpServer := &http.Server{Addr: self.address}

  go func() {
    if err := httpServer.ListenAndServe(); err != nil {
      log.Fatal(err)
    }
  }()

  <-stop
  
  ctx, cancel := context.WithTimeout(context.Background(), 5 * time.Second)
  defer cancel()

  if err := httpServer.Shutdown(ctx); err != nil {
      log.Println("HTTP shutdown error:", err)
  }
  if err := httpServer.Close(); err != nil {
      log.Println("HTTP server close error:", err)
  }
  log.Println("Server stopped gracefully.")
}

func (self *Server) loadCleanBatch() *Batch {
  return &Batch{
    clean: CB2GB(C.getCleanBatch()), 
    adv: nil,
  }
}

func (self *Server) onGetAttack(w http.ResponseWriter, r *http.Request) { 
  self.attackMutex.RLock()
  w.Write(self.attackData)
  self.attackMutex.RUnlock()
  
  w.Flush()
}

func (self *Server) onPostAttack(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Fatal("Could not read request body:", err)
  }

  self.attackMutex.Lock()
  self.attackData = data
  self.attackID += 1
  self.attackMutex.Unlock()

  //TODO: Handle the case when the attack is updated during the training!
  // Resend batches that are currently being worked on or do not bother?
}

func (self *Server) onGetModel(w http.ResponseWriter, r *http.Request) {
  self.modelMutex.RLock()
  w.Write(self.modelData)
  self.modelMutex.RUnlock()

  w.Flush()
}

func (self *Server) onPostModel(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Fatal("Could not read request body:", err)
  }

  self.modelMutex.Lock()
  self.modelData = data[1:]
  self.modelID += 1
  self.modelMutex.Unlock()

  if data[0] == 1 {
    //TODO: Handle the case when not just the parameters of the model were changed,
    // but the entire architecture changed.
    // Resend batches that are currently being worked on?
  }

  // Move every expired batch back to the freeQ, to redo them.
  self.workQ.Range(func(batch any, startModelID any) bool {
    if self.modelID - startModelID.(uint64) > self.maxPatiente {
      // After a very long time, the subtruction can cause a slight bug when the
      // startModelID is somewhere at the end of the uint64 range and the modelID
      // fliped already to the beginning of the calue range.
      self.workQ.Delete(batch)
      self.freeQ <- batch.(*Batch)
    }
    return true
  })
}

func (self *Server) onGetAdvBatch(w http.ResponseWriter, r *http.Request) {
  //TODO: Check if doneQ is nil or not. If it is nil, the user tried to iterate without
  // setting up the server properly.
  w.Write((<-self.doneQ).adv)
  w.Flush()
}

func (self *Server) onPostAdvBatch(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Fatal("Could not read request body:", err)
  }

  // Read the first 8 bytes of the data as the memory address where the
  // batch is stored. This is basically a batch ID.
  batch := (*Batch)(unsafe.Pointer(&data[0]))

  // If the batch was already moved back to the freeQ, just drop the batch.
  if _, ok := self.workQ.LoadAndDelete(batch); !ok { 
    return
  }
  
  batch.clean = nil
  batch.adv = data[8:]
  self.doneQ <- batch
}

func (self *Server) onGetCleanBatch(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onGetIDs(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onGetNumBatches(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onPostDataset(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onPostDataloader(w http.ResponseWriter, r *http.Request) {

}

func main() {
  s := InitServer(":8080")
  s.Run()
}
