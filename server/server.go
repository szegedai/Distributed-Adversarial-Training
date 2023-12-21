package main

/*
#include "pyinterface.h"
*/
import "C"
import (
  "fmt"
  "net/http"
  . "github.com/Jcowwell/go-algorithm-club/Heap"
  . "github.com/Jcowwell/go-algorithm-club/Utils"
)

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
  dispathRequest("/attack", self.onGetAttack, self.onPostAttack)
  dispathRequest("/model", self.onGetModel, self.onPostModel)
  dispathRequest("/adv_batch", self.onGetAdvBatch, self.onPostAdvBatch)
  dispathRequest("/clean_batch", self.onGetCleanBatch, nil)
  dispathRequest("/ids", self.onGetIDs, nil)
  dispathRequest("/num_batches", self.onGetNumBatches, nil)
  dispathRequest("/dataset", nil, self.onPostDataset)
  dispathRequest("/dataloader", nil, self.onPostDataloader)

  err := http.ListenAndServe(self.address, nil)
  if err != nil {
    panic(err)
  }
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
