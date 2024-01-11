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
  "encoding/binary"
  "flag"
  "fmt"
)

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

type BatchMeta struct {
  Batch *Batch
  TimeStamp uint64
}

type Batch struct {
  ID uint64
  Clean []byte
  Adv []byte
}

type Server struct { 
  address string

  queueLimit uint64
  maxPatiente uint64

  modelData []byte
  modelID uint64
  modelMutex sync.RWMutex
  attackData []byte
  attackID uint64
  attackMutex sync.RWMutex
  nextBatchID uint64
  nextBatchIDMutex sync.Mutex

  freeQ chan *Batch
  workQ sync.Map
  doneQ chan *Batch

  setupWG sync.WaitGroup
  finishDataSetup func()
  finishParametersSetup func()
  finishAttackSetup func()
  finishModelSetup func()
}

func (self *Server) Reset() {
  self.queueLimit = 0
  self.maxPatiente = 0
  self.modelData = nil 
  self.modelID = 0
  self.attackData = nil
  self.attackID = 0
  self.nextBatchID = 0
  self.freeQ = nil
  self.workQ = sync.Map{}
  self.doneQ = nil
  self.setupWG = sync.WaitGroup{}
  self.finishDataSetup = sync.OnceFunc(func() {
    self.setupWG.Done()
  })
  self.finishParametersSetup = sync.OnceFunc(func() {
    self.setupWG.Done()
  })
  self.finishAttackSetup = sync.OnceFunc(func() {
    self.setupWG.Done()
  })
  self.finishModelSetup = sync.OnceFunc(func() {
    self.setupWG.Done()
  })

  self.setupWG.Add(4)
}

func (self *Server) Run() {
  stop := make(chan os.Signal)
  signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

  C.initPython()
  defer C.finalizePython()

  dispathRequest("/attack", self.onGetAttack, self.onPostAttack)
  dispathRequest("/model", self.onGetModel, self.onPostModel)
  dispathRequest("/adv_batch", self.onGetAdvBatch, self.onPostAdvBatch)
  dispathRequest("/clean_batch", self.onGetCleanBatch, nil)
  dispathRequest("/ids", self.onGetIDs, nil)
  dispathRequest("/num_batches", self.onGetNumBatches, nil)
  dispathRequest("/data", nil, self.onPostData)
  dispathRequest("/parameters", nil, self.onPostParameters)
  http.HandleFunc("/reset", func(w http.ResponseWriter, r *http.Request) {
    log.Println("Reseting server!")
    self.Reset()
  })

  httpServer := &http.Server{Addr: self.address}

  go func() {
    if err := httpServer.ListenAndServe(); err != nil {
      log.Fatal(err)
    }
  }()

  log.Println("Server started with config: { Address:", self.address, "}")

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

func (self *Server) loadCleanBatch() {
  self.nextBatchIDMutex.Lock()

  batch := &Batch{
    ID: self.nextBatchID,
    Clean: CB2GB(C.getCleanBatch()), 
    Adv: nil,
  }

  self.nextBatchID += 1
  self.nextBatchIDMutex.Unlock()

  self.freeQ <- batch
}

func (self *Server) onGetAttack(w http.ResponseWriter, r *http.Request) {
  self.setupWG.Wait()

  self.attackMutex.RLock()
  w.Write(self.attackData)
  self.attackMutex.RUnlock()
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

  self.finishAttackSetup()

  log.Println("Attack updated")
}

func (self *Server) onGetModel(w http.ResponseWriter, r *http.Request) {
  self.setupWG.Wait()

  self.modelMutex.RLock()
  w.Write(self.modelData)
  self.modelMutex.RUnlock()
}

func (self *Server) onPostModel(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Fatal("Could not read request body:", err)
  }

  self.modelMutex.Lock()
  self.modelData = data
  self.modelID += 1
  self.modelMutex.Unlock()

  // Move every expired batch back to the freeQ, to redo them.
  self.workQ.Range(func(batchID any, batchMeta any) bool {
    self.modelMutex.RLock()
    if self.modelID - batchMeta.(BatchMeta).TimeStamp > self.maxPatiente {
      // After a very long time, the subtruction can cause a slight bug when the
      // startModelID is somewhere at the end of the uint64 range and the modelID
      // fliped already to the beginning of the calue range.
      self.workQ.Delete(batchID)
      self.freeQ <- batchMeta.(BatchMeta).Batch

      log.Println("Batch", batchID, "has expired")
    }
    self.modelMutex.RUnlock()
    return true
  })

  self.finishModelSetup()

  log.Println("Model updated")
}

func (self *Server) onGetAdvBatch(w http.ResponseWriter, r *http.Request) {
  self.setupWG.Wait()

  w.Write((<-self.doneQ).Adv)
}

func (self *Server) onPostAdvBatch(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Fatal("Could not read request body:", err)
  }

  batchID := binary.BigEndian.Uint64(data[0:8])

  // If the batch was already moved back to the freeQ, just drop the batch.
  batchMeta, loaded := self.workQ.LoadAndDelete(batchID)
  if !loaded { 
    return
  }
  batch := batchMeta.(BatchMeta).Batch
  
  batch.Clean = nil
  batch.Adv = data[8:]
  self.doneQ <- batch
}

func (self *Server) onGetCleanBatch(w http.ResponseWriter, r *http.Request) {
  self.setupWG.Wait()

  batch := <-self.freeQ

  go self.loadCleanBatch()
  
  self.modelMutex.RLock()
  self.workQ.Store(batch.ID, BatchMeta{batch, self.modelID})
  self.modelMutex.RUnlock()

  batchIDBytes := make([]byte, 8)
  binary.BigEndian.PutUint64(batchIDBytes, batch.ID)

  w.Write(batchIDBytes)
  w.Write(batch.Clean)
}

func (self *Server) onGetIDs(w http.ResponseWriter, r *http.Request) {
  self.setupWG.Wait()
  
  attackIDBytes := make([]byte, 8)
  modelIDBytes := make([]byte, 8)
  binary.BigEndian.PutUint64(attackIDBytes, self.attackID)
  binary.BigEndian.PutUint64(modelIDBytes, self.modelID)

  w.Write(attackIDBytes)
  w.Write(modelIDBytes)
}

func (self *Server) onGetNumBatches(w http.ResponseWriter, r *http.Request) {
  self.setupWG.Wait()

  numBatchesBytes := make([]byte, 8)
  binary.BigEndian.PutUint64(numBatchesBytes, uint64(C.getNumBatches()))

  w.Write(numBatchesBytes)
}

func (self *Server) onPostData(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)

  if err != nil {
    log.Fatal("Could not read request body:", err)
  }
  
  C.updateData(GB2CB(data))

  var i uint64
  for i = 0; i < self.queueLimit; i++ {
    self.loadCleanBatch()
  }

  self.finishDataSetup()
}

func (self *Server) onPostParameters(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Fatal("Could not read request body:", err)
  }

  self.maxPatiente = binary.BigEndian.Uint64(data[0:8])
  self.queueLimit = binary.BigEndian.Uint64(data[8:16])

  self.freeQ = make(chan *Batch, self.queueLimit)
  self.doneQ = make(chan *Batch, self.queueLimit)

  self.finishParametersSetup()

  log.Println("Parameters updated: { MaxPatiente:", self.maxPatiente, "QueueLimit:", self.queueLimit, "}")
}

func main() {
  address := flag.String("A", ":8080", "Address and port for the server to listen on")
  flag.Parse()

  s := Server{address: *address}
  s.Reset()
  s.Run()
}
