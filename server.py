import torch
import threading
import bottle
import time
import dill
import argparse
from heapq import heappush, heappop


class Server:
    # TODO:
    # 2) Handle the case when a request comes in before the dataset and dataloader objects are initialised!
    # 3) Handle the case when dataloader constructuion is called before dataset construction!
    def __init__(self, port):
        self._port = int(port)

        self._attack_info = None
        self._latest_attack_id = -1
        self._attack_mutex = threading.Lock()
        self._model = None
        self._latest_model_id = -1
        self._model_mutex = threading.Lock()

        self._dataset = None
        self._dataloader = None
        self._dataloader_iter = None
        self._data_mutex = threading.Lock()
        self._queue_soft_limit = None

        self._batch_store = dict()
        self._old_batch_store = dict()  # stores the unperturbed batches in case the batches in the done queue has to be redone.
        self._batch_store_mutex = threading.Lock()
        self._old_batch_sotre_mutex = threading.Lock()

        self._free_q = list()  # Need priority so batches that need redistribution, get to the top.
        self._working_q = dict()  # Fast access when searching for specific batch id and iterable to check for timeouts (this needs a lock).
        self._done_q = list()  # Need priority so batches that were redistributed, get to the top.
        self._free_q_mutex = threading.Lock()
        self._working_q_mutex = threading.Lock()
        self._done_q_mutex = threading.Lock()

        self._latest_unused_batch_id = 0
        self._max_patiente = None

        self._timeout_handler_thread = threading.Thread(target=self._run_timeout_handler, daemon=True)
        
    def run(self):
        self._timeout_handler_thread.start()
        self._run_request_handler()

    def _load_batch(self):
        with self._data_mutex, self._batch_store_mutex, self._free_q_mutex:
            try:
                batch = next(self._dataloader_iter)
            except StopIteration:
                self._dataloader_iter = iter(self._dataloader)
                batch = next(self._dataloader_iter)
            id = self._latest_unused_batch_id
            self._latest_unused_batch_id += 1
            self._batch_store[id] = batch
            heappush(self._free_q, id)

    def _run_request_handler(self):
        bottle.get('/attack')(self._on_get_attack)
        bottle.post('/attack')(self._on_post_attack)
        bottle.get('/clean_batch')(self._on_get_clean_batch)
        bottle.get('/adv_batch')(self._on_get_adv_batch)
        bottle.post('/adv_batch')(self._on_post_adv_batch)
        bottle.get('/ids')(self._on_get_ids)
        bottle.get('/model')(self._on_get_model)
        bottle.post('/model')(self._on_post_model)
        bottle.post('/dataset')(self._on_post_dataset)
        bottle.post('/dataloader')(self._on_post_dataloader)
        bottle.get('/num_batches')(self._on_get_num_batches)

        # TODO: Change the bottle server backend from "wsgiref" to something else that is multithreaded!
        bottle.run(host='0.0.0.0', port=self._port, quiet=False, debug=True, server='wsgiref')
        #bottle.run(host='0.0.0.0', port=self._port, quiet=True, server='wsgiref')

    def _run_timeout_handler(self):
        while True:
            time.sleep(2)
            # Check if working queue has batches that run out of time.
            # If a batch is expired, pop it from the working queue and add it to the free queue.
            with self._working_q_mutex, self._free_q_mutex, self._attack_mutex:
                for batch_id in list(self._working_q.keys()):
                    if self._working_q[batch_id] - self._latest_model_id > self._max_patiente:
                        del self._working_q[batch_id]
                        heappush(self._free_q, batch_id)

    def _on_get_attack(self):
        # Send the current attack class and arguments.
        with self._attack_mutex:
            return dill.dumps(self._attack_info)

    def _on_post_attack(self):
        # Override the current attack class and args with the recieved one.
        attack_info = dill.loads(bottle.request.body.read())
        with self._attack_mutex, self._free_q_mutex, self._working_q_mutex, self._done_q_mutex:
            self._attack_info = attack_info
            self._latest_attack_id += 1

            # Move all batches to the free queue to resend them.
            for batch_id in self._working_q.keys():
                heappush(self._free_q, batch_id)
            for batch_id in self._done_q:
                heappush(self._free_q, batch_id)
            self._working_q.clear()
            self._done_q.clear()
            
    def _on_get_clean_batch(self):
        # Pop a clean batch from the free queue, move it to the working queue and send the batch id and the batch itself.
        with self._free_q_mutex, self._working_q_mutex, self._batch_store_mutex, self._model_mutex:
            if not len(self._free_q):
                # There are no clean batches available, probably because the full initialisation process is not yet finished.
                bottle.response.status = 204
                return
            batch_id = heappop(self._free_q)
            self._working_q[batch_id] = self._latest_model_id
            return dill.dumps([batch_id, self._batch_store[batch_id]])

    def _on_get_adv_batch(self):
        # Pop a batch id from the done queue, remove it from the batch store and send it.
        with self._done_q_mutex, self._batch_store_mutex, self._old_batch_sotre_mutex:
            if not len(self._done_q):
                # There are no adversarial batches available.
                # This can happen if the initialisation process is not yet finished or 
                # if the nodes have not yet had enough time to generate new adversarial batches.
                bottle.response.status = 204
                return
            batch_id = heappop(self._done_q)
            del self._old_batch_store[batch_id]
            return dill.dumps(self._batch_store.pop(batch_id))

    def _on_post_adv_batch(self):
        # Move the recieved batch id from the woring queue to the done queue and override the batch in the batch store.
        # If done queue reached soft limit, drop recieved batch and move it to the free queue.
        # If the recieved batch is not in the working queue, drop it and do nothing.
        # If the batch has expired, remove it from the working queue and add it to the free queue.
        batch_id, batch = dill.loads(bottle.request.body.read())
        with self._done_q_mutex, self._working_q_mutex:
            if not (batch_id in self._working_q):
                return
        with self._done_q_mutex, self._working_q_mutex, self._free_q_mutex, self._model_mutex:
            if self._working_q[batch_id] - self._latest_model_id > self._max_patiente or len(self._done_q) >= self._queue_soft_limit:
                del self._working_q[batch_id]
                heappush(self._free_q, batch_id)
                return
        with self._done_q_mutex, self._working_q_mutex, self._batch_store_mutex, self._old_batch_sotre_mutex:
            del self._working_q[batch_id]
            self._old_batch_store[batch_id] = self._batch_store[batch_id]
            self._batch_store[batch_id] = batch
            heappush(self._done_q, batch_id)
        self._load_batch()

    def _on_get_ids(self):
        # Send the latest attack and model ids.
        with self._attack_mutex, self._model_mutex:
            if self._latest_attack_id == -1 and self._latest_model_id == -1:
                # The server has not yet finished the initialisation process.
                bottle.response.status = 204
                return
            return dill.dumps([self._latest_attack_id, self._latest_model_id])

    def _on_get_model(self):
        with self._model_mutex:
            return dill.dumps(self._model)

    def _on_post_model(self):
        with self._model_mutex, self._free_q_mutex, self._working_q_mutex, self._done_q_mutex: 
            self._model, new_architecture = dill.loads(bottle.request.body.read())
            self._latest_model_id += 1

            if new_architecture:
                # Move all batches to the free queue to resend them.
                for batch_id in self._working_q.keys():
                    heappush(self._free_q, batch_id)
                for batch_id in self._done_q:
                    heappush(self._free_q, batch_id)
                self._working_q.clear()
                self._done_q.clear()

    def _on_post_dataset(self):
        # Construct the dataset object using the recieved class and arguments.
        with self._data_mutex, self._free_q_mutex, self._working_q_mutex, self._done_q_mutex:
            dataset_class, args, kwargs = dill.loads(bottle.request.body.read())
            self._dataset = dataset_class(*args, **kwargs)
            
            # Clear all queues and dataloader. The user have to update the dataloader to be able to start working.
            self._free_q.clear()
            self._working_q.clear()
            self._done_q.clear()
            self._dataloader = None
            self._dataloader_iter = None

    def _on_post_dataloader(self):
        # Construct the dataloader object using the dataset object and the recieved arguments.
        # Clear the batch store and all queues since a new dataloader was given so the ongoing and done batches are irrelevant.
        with self._data_mutex, self._batch_store_mutex, self._free_q_mutex, self._working_q_mutex, self._done_q_mutex:
            if not self._dataset:
                # There was no dataset to use for the dataloader.
                bottle.response.status = 204
                return
            self._free_q.clear()
            self._working_q.clear()
            self._done_q.clear()
            self._batch_store.clear()
            args, kwargs, self._max_patiente, self._queue_soft_limit = dill.loads(bottle.request.body.read())
            self._dataloader = torch.utils.data.DataLoader(self._dataset, *args, **kwargs)
            self._dataloader_iter = iter(self._dataloader)
        # Preload batches as fast as possible to not starve the nodes on startup.
        for _ in range(self._queue_soft_limit):
            self._load_batch()

    def _on_get_num_batches(self):
        with self._data_mutex:
            if not self._dataloader:
                # There was no dataloader to use.
                bottle.response.status = 204
                return
            return dill.dumps(len(self._dataloader))


if __name__ == '__main__':
    # Parse arguments and start the server.
    parser = argparse.ArgumentParser(
        prog='Server',
        description='Execution server for distributed adversarial training.'
    )
    parser.add_argument(
        'port',
        nargs='?',
        type=int,
        default=8080,
        help='Port number to use for communication.'
    )
    args = vars(parser.parse_args())
    
    Server(**args).run()

