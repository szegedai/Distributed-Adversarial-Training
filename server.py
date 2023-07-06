import torch
import threading
import bottle
import time
import pickle
from heapq import heappush, heappop


class Server:
    # TODO:
    # 1) Handle data loading into the free queue!
    # 2) Handle the case when a request comes in before the dataset and dataloader objects are initialised!
    # 3) Handle the case when dataloader constructuion is called before dataset construction!
    def __init__(self, port, max_patiente=300):
        self._port = int(port)

        self._attack = None
        self._attack_mutex = threading.Lock()
        self._latest_model_id = 0

        self._dataset = None
        self._dataloader = None
        self._dataloader_iter = None
        self._data_mutex = threading.Lock()
        self._queue_soft_limit = None

        self._batch_store = dict()
        self._batch_store_mutex = threading.Lock()

        self._free_q = list()  # Need priority so batches that need redistribution, get to the top.
        self._working_q = dict()  # Fast access when searching for specific batch id and iterable to check for timeouts (this needs a lock).
        self._done_q = list()  # Need priority so batches that were redistributed, get to the top.
        self._free_q_mutex = threading.Lock()
        self._working_q_mutex = threading.Lock()
        self._done_q_mutex = threading.Lock()

        self._latest_unused_batch_id = 0
        self._max_patiente = max_patiente

        self._dataloader_thread = threading.Thread(target=self._run_dataloder, daemon=True)
        self._timeout_handler_thread = threading.Thread(target=self._run_timeout_handler, daemon=True)
        
    def run(self):
        self._dataloader_thread.start()
        self._timeout_handler_thread.start()
        self._run_request_handler()

    def _load_batch(self):
        with self._data_mutex, self._batch_store_mutex, self._free_q_mutex:
            if len(self._free_q) < self._queue_soft_limit:
                try:
                    batch = next(self._dataloader_iter)
                except StopIteration:
                    self._dataloader_iter = iter(self._dataloader)
                    batch = next(self._dataloader_iter)
                id = self._latest_unused_batch_id
                self._latest_unused_batch_id += 1
                self._batch_store[id] = batch
                heappush(self._free_q, id)


    def _run_dataloder(self):
        # Keep the free queue filled.
        while True:
            time.sleep(1)  # This is to not kepp the cpu core utulisation on max.
            if self._dataloader_iter:
                self._load_batch()

    def _run_request_handler(self):
        bottle.get(self._on_get_attack, '/attack')
        bottle.post(self._on_post_attack, '/attack')
        bottle.get(self._on_get_clean_batch, '/clean_batch')
        bottle.get(self._on_get_adv_batch, '/adv_batch')
        bottle.post(self._on_post_adv_batch, '/adv_batch')
        bottle.get(self._on_get_model_id, '/model_id')
        bottle.get(self._on_get_model_state, '/model_state')
        bottle.post(self._on_post_model_state, '/model_state')
        bottle.post(self._on_post_dataset, '/dataset')
        bottle.post(self._on_post_dataloader, '/dataloader')

        # TODO: Change the bottle server backend from "wsgiref" to something else that is multithreaded!
        bottle.run(host='0.0.0.0', port=self._port, quiet=True, server='wsgiref')

    def _run_timeout_handler(self):
        while True:
            time.sleep(1)
            # Check if working queue has batches that run out of time.
            # If a batch is expired, pop it from the working queue and add it to the free queue.
            with self._working_q_mutex, self._free_q_mutex:
                for batch_id in list(self._working_q.keys()):
                    if self._working_q[batch_id] - time.time() > self._max_patiente:
                        del self._working_q[batch_id]
                        heappush(self._free_q, batch_id)

    def _on_get_attack(self):
        # Send the current attack object. (It includes the current model object as well.)
        with self._attack_mutex:
            return pickle.dumps(self._attack)

    def _on_post_attack(self):
        # Override the current attack object with the recieved one. (It includes the new model object as well.)
        # Check if attack has "model" attribute and "perturb" method!
        attack = pickle.loads(bottle.request.POST('attack'))
        assert hasattr(attack, 'model'), 'The attack object must have an attribute named "model"!'
        assert hasattr(attack, 'perturb') and callable(attack.perturb), 'The attack object must have a method named "perturb"!'
        with self._attack_mutex:
            self._attack = attack
            # Maybe add: self._attack.model.cpu() ???

    def _on_get_clean_batch(self):
        # Pop a clean batch from the free queue, move it to the working queue and send the batch id and the batch itself.
        with self._free_q_mutex, self._working_q_mutex, self._batch_store_mutex:
            batch_id = heappop(self._free_q)
            self._working_q[batch_id] = time.time()
            return pickle.dumps([batch_id, self._batch_store[batch_id]])

    def _on_get_adv_batch(self):
        # Pop a batch id from the done queue, remove it from the batch store and send it.
        with self._done_q_mutex, self._batch_store_mutex:
            batch_id = heappop(self._done_q)
            return pickle.dumps(self._batch_store.pop(batch_id))

    def _on_post_adv_batch(self):
        # Move the recieved batch id from the woring queue to the done queue and override the batch in the batch store.
        # If done queue reached soft limit, drop recieved batch and move it to the free queue.
        # If the recieved batch is not in the working queue, drop it and do nothing.
        # It is not needed to check if the batch is expired, since in the worst case scenario, at max 1 sec has 
        # passed since the last check and getting 1 sec over the max patiente is acceptable.
        batch_id = pickle.loads(bottle.request.POST['batch_id'])
        batch = pickle.loads(bottle.request.POST['batch'])
        with self._done_q_mutex, self._working_q_mutex:
            if not (batch_id in self._working_q and len(self._done_q) < self._queue_soft_limit):
                return
        with self._done_q_mutex, self._working_q_mutex, self._batch_store_mutex:
            del self._working_q[batch_id]
            self._batch_store[batch_id] = batch
            heappush(self._done_q, batch_id)

    def _on_get_model_id(self):
        # Send the latest model id.
        with self._attack_mutex:
            return pickle.dumps(self._latest_model_id)

    def _on_get_model_state(self):
        # Send the state_dict of the latest model.
        with self._attack_mutex:
            return pickle.dumps(self._attack.model.state_dict())

    def _on_post_model_state(self):
        # Update the current model with the recieved new model state_dict.
        with self._attack_mutex:
            state = pickle.loads(bottle.request.POST['model_state'])
            self._attack.model.load_state_dict(state)
            self._latest_model_id += 1

    def _on_post_dataset(self):
        # Construct the dataset object using the recieved class and arguments.
        with self._data_mutex:
            dataset_class = pickle.loads(bottle.request.POST['dataset_class'])
            kwargs = pickle.loads(bottle.request.POST['kwargs'])
            self._dataset = dataset_class(**kwargs)

    def _on_post_dataloader(self):
        # Construct the dataloader object using the dataset object and the recieved arguments.
        # Clear the batch store and all queues since a new dataloader was given so the ongoing and done batches are irrelevant.
        with self._data_mutex, self._batch_store_mutex, self._free_q_mutex, self._working_q_mutex, self._done_q_mutex:
            self._free_q.clear()
            self._working_q.clear()
            self._done_q.clear()
            self._batch_store.clear()
            kwargs = pickle.loads(bottle.request.POST['kwargs'])
            self._dataloader = torch.utils.data.DataLoader(**kwargs)
            self._dataloader_iter = iter(self._dataloader)
        # Preload batches as fast as possible to not starve the nodes on startup.
        for _ in range(self._queue_soft_limit):
            self._load_batch()



if __name__ == '__main__':
    # TODO: Parse arguments and start the server.
    Server('8080').run()

