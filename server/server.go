package main

/*
#include "pyinterface.h"
*/
import "C"
import (
	"encoding/binary"
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

const (
  SETUP_ATTACK = iota
  SETUP_MODEL
  SETUP_MODEL_STATE
  SETUP_DATASET
  SETUP_DATALOADER
  SETUP_PARAMETERS
)

type TODOSync struct {
  todos []sync.Once
  doneWG sync.WaitGroup
  doneCount int
  mutex sync.Mutex
}

func (self *TODOSync) Init(length int) {
  self.mutex = sync.Mutex{}

  self.mutex.Lock()

  self.doneWG = sync.WaitGroup{}
  self.doneWG.Add(length)
  self.doneCount = length
  self.todos = make([]sync.Once, length)

  self.mutex.Unlock()
}

func (self *TODOSync) Reset() {
  self.mutex.Lock()

  length := len(self.todos)
  self.doneWG.Add(length - self.doneCount)
  self.doneCount = length
  self.todos = make([]sync.Once, length)

  self.mutex.Unlock()
}

func (self *TODOSync) Done(idx int) {
  self.todos[idx].Do(func() {
    self.mutex.Lock()

    self.doneWG.Done()
    self.doneCount--

    self.mutex.Unlock()
  })
}

func (self *TODOSync) Wait() {
  self.doneWG.Wait()
}

type BatchMeta struct {
  Batch *Batch
  TimeStamp uint64
}

type Batch struct {
  ID uint64
  Clean []byte
  Adv []byte
  ExtraData *string
}

type Server struct {
  address string

  queueLimit uint64
  maxPatiente uint64

  modelData []byte
  modelID uint64
  modelStateData []byte
  modelStateID uint64
  modelMutex sync.RWMutex
  attackData []byte
  attackID uint64
  attackMutex sync.RWMutex
  nextBatchID uint64
  nextBatchIDMutex sync.Mutex

  freeQ chan *Batch
  workQ sync.Map
  doneQ chan *Batch

  setup *TODOSync
}

func (self *Server) Reset() {
  self.queueLimit = 0
  self.maxPatiente = 0
  self.modelData = nil
  self.modelID = 0
  self.modelStateData = nil
  self.modelStateID = 0
  self.attackData = nil
  self.attackID = 0
  self.nextBatchID = 0

  if self.freeQ != nil {
    //close(self.freeQ)
    empty := false
    for !empty {
      select {
      case <-self.freeQ:
      default:
         empty = true
      }
    }
  }
  if self.doneQ != nil {
    //close(self.doneQ)
    empty := false
    for !empty {
      select {
      case <-self.doneQ:
      default:
         empty = true
      }
    }
  }
  self.freeQ = nil
  self.workQ = sync.Map{}
  self.doneQ = nil

  if self.setup == nil {
    self.setup = &TODOSync{}
    self.setup.Init(6)
  }
  self.setup.Reset()
}

func (self *Server) Run() {
  log.Println("Starting server with config: { Address:", self.address, "}")

  stop := make(chan os.Signal)
  signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

  C.initPython()
  defer C.finalizePython()

  dispathRequest("/attack", self.onGetAttack, self.onPostAttack)
  dispathRequest("/model", self.onGetModel, self.onPostModel)
  dispathRequest("/model_state", self.onGetModelState, self.onPostModelState)
  dispathRequest("/adv_batch", self.onGetAdvBatch, self.onPostAdvBatch)
  dispathRequest("/clean_batch", self.onGetCleanBatch, nil)
  dispathRequest("/ids", self.onGetIDs, nil)
  dispathRequest("/num_batches", self.onGetNumBatches, nil)
  dispathRequest("/dataset", nil, self.onPostDataset)
  dispathRequest("/dataloader", nil, self.onPostDataloader)
  dispathRequest("/parameters", nil, self.onPostParameters)
  http.HandleFunc("/reset", func(w http.ResponseWriter, r *http.Request) {
    log.Println("Reseting server")
    self.Reset()
  })

  httpServer := &http.Server{Addr: self.address}

  go func() {
    if err := httpServer.ListenAndServe(); err != nil {
      log.Fatalln(err)
    }
  }()

  log.Println("Ready and running")

  <-stop
 }

func (self *Server) loadCleanBatch() {
  self.nextBatchIDMutex.Lock()

  batch := &Batch{
    ID: self.nextBatchID,
    Clean: CB2GB(C.getCleanBatch()),
    Adv: nil,
    ExtraData: nil,
  }

  self.nextBatchID += 1
  self.nextBatchIDMutex.Unlock()

  self.freeQ <- batch
}

func (self *Server) onGetAttack(w http.ResponseWriter, r *http.Request) {
  self.setup.Wait()

  self.attackMutex.RLock()
  w.Write(self.attackData)
  self.attackMutex.RUnlock()
}

func (self *Server) onPostAttack(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Println(err)
    return
  }

  self.attackMutex.Lock()
  self.attackData = data
  self.attackID += 1
  self.attackMutex.Unlock()

  //TODO: Handle the case when the attack is updated during the training!
  // Resend batches that are currently being worked on or do not bother?

  self.setup.Done(SETUP_ATTACK)
}

func (self *Server) onGetModel(w http.ResponseWriter, r *http.Request) {
  self.setup.Wait()

  self.modelMutex.RLock()
  w.Write(self.modelData)
  self.modelMutex.RUnlock()
}

func (self *Server) onPostModel(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Println(err)
    return
  }

  self.modelMutex.Lock()
  self.modelData = data
  self.modelID += 1
  self.modelMutex.Unlock()

  self.setup.Done(SETUP_MODEL)

  log.Println("Model architecture updated")
}

func (self *Server) onGetModelState(w http.ResponseWriter, r *http.Request) {
  self.setup.Wait()

  self.modelMutex.RLock()
  w.Write(self.modelStateData)
  self.modelMutex.RUnlock()
}

func (self *Server) onPostModelState(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Println(err)
    return
  }

  self.modelMutex.Lock()
  self.modelStateData = data
  self.modelStateID += 1
  self.modelMutex.Unlock()

  // Resend expired batches.
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

  self.setup.Done(SETUP_MODEL_STATE)
}

func (self *Server) onGetAdvBatch(w http.ResponseWriter, r *http.Request) {
  self.setup.Wait()

  batch := <-self.doneQ

  w.Header().Add("X-Extra-Data", *batch.ExtraData)
  w.Write(batch.Adv)
}

func (self *Server) onPostAdvBatch(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  extraData := r.Header.Get("X-Extra-Data")
  if err != nil {
    log.Println(err)
    return
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
  if extraData != "" {
    batch.ExtraData = &extraData
  }
  self.doneQ <- batch
}

func (self *Server) onGetCleanBatch(w http.ResponseWriter, r *http.Request) {
  self.setup.Wait()

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
  self.setup.Wait()

  attackIDBytes := make([]byte, 8)
  modelIDBytes := make([]byte, 8)
  modelStateIDBytes := make([]byte, 8)
  binary.BigEndian.PutUint64(attackIDBytes, self.attackID)
  binary.BigEndian.PutUint64(modelIDBytes, self.modelID)
  binary.BigEndian.PutUint64(modelStateIDBytes, self.modelStateID)

  w.Write(attackIDBytes)
  w.Write(modelIDBytes)
  w.Write(modelStateIDBytes)
}

func (self *Server) onGetNumBatches(w http.ResponseWriter, r *http.Request) {
  self.setup.Wait()

  numBatchesBytes := make([]byte, 8)
  binary.BigEndian.PutUint64(numBatchesBytes, uint64(C.getNumBatches()))

  w.Write(numBatchesBytes)
}

func (self *Server) onPostDataset(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Println(err)
    return
  }
  
  C.updateDataset(GB2CB(data))

  self.setup.Done(SETUP_DATASET)

  log.Println("Dataset updated")
}

func (self *Server) onPostDataloader(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Println(err)
    return
  }
  
  C.updateDataloader(GB2CB(data))

  var i uint64
  for i = 0; i < self.queueLimit; i++ {
    self.loadCleanBatch()
  }

  self.setup.Done(SETUP_DATALOADER)

  log.Println("Dataloader updated")
}

func (self *Server) onPostParameters(w http.ResponseWriter, r *http.Request) {
  data, err := ioutil.ReadAll(r.Body)
  if err != nil {
    log.Println(err)
    return
  }

  self.maxPatiente = binary.BigEndian.Uint64(data[0:8])
  self.queueLimit = binary.BigEndian.Uint64(data[8:16])

  self.freeQ = make(chan *Batch, self.queueLimit)
  self.doneQ = make(chan *Batch, self.queueLimit)

  self.setup.Done(SETUP_PARAMETERS)

  log.Println("Parameters updated: { MaxPatiente:", self.maxPatiente, "QueueLimit:", self.queueLimit, "}")
}

func main() {
  address := flag.String("A", ":8080", "Address and port for the server to listen on")
  flag.Parse()

  s := Server{address: *address}
  s.Reset()
  s.Run()
}
