package main

/*
#include "pyinterface.h"
*/
import "C"
import (
	"fmt"
	"log"
	"net/http"
	"reflect"
	"unsafe"
  "os"
  "os/signal"
  "syscall"
  "context"
  "time"
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
        getHandler(w, r)
      } else {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
      }
    case http.MethodPost:
      if postHandler != nil {
        postHandler(w, r)
      } else {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
      } 
    default:
      http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
  })
}

type Server struct {
  address string

  queueSoftLimit uint64
  maxPatiente uint64

  modelData []byte
  modelID uint64
  attackData []byte
  attackID uint64

  freeQ Heap[uint64]
  workingQ map[uint64]uint64
  doneQ Heap[uint64]

  batchStore map[uint64][]byte
  oldBatchStore map[uint64][]byte

  latestUnusedBatchID uint64
}

func InitServer(address string) *Server {
  return &Server{
    address, 
    0, 0, 
    nil, 0, 
    nil, 0, 
    Heap[uint64]{
      []uint64{}, 
      LessThan[uint64],
    }, 
    make(map[uint64]uint64), 
    Heap[uint64]{
      []uint64{},
      LessThan[uint64],
    },
    make(map[uint64][]byte),
    make(map[uint64][]byte),
    1,
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

func (self *Server) loadCleanBatch() []byte {
  return CB2GB(C.getCleanBatch())
}

func (self *Server) onGetAttack(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onPostAttack(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onGetModel(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onPostModel(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onGetAdvBatch(w http.ResponseWriter, r *http.Request) {

}

func (self *Server) onPostAdvBatch(w http.ResponseWriter, r *http.Request) {

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
