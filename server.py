import threading
import queue
import bottle
import time


class Server:
    def __init__(self, port):
        self._port = port
        self._attack = None
        self._dataset = None
        self._dataloader = None
        self._queue_soft_limit = None
        self._batch_store = dict()
        self._free_q = queue.PriorityQueue()  # Need priority so batches that need redistribution, get to the top.
        self._working_q = dict()  # Fast access when searching for specific batch id and iterable to check for timeouts (this needs a lock).
        self._done_q = queue.PriorityQueue()  # Need priority so batches that were redistributed, get to the top.
        self._working_q_mutex = threading.Lock()
        self._latest_unused_id = 0

        self._request_handler_thread = threading.Thread(target=self._run_request_handler, daemon=True)
        self._timeout_handler_thread = threading.Thread(target=self._run_timeout_handler, daemon=True)
        
    def run(self):
        self._request_handler_thread.start()
        self._timeout_handler_thread.start()

        # The program only gets through any of the following two lines if an unexpected error is raised on the threads.
        # TODO: Handle errors!
        self._request_handler_thread.join()
        self._timeout_handler_thread.join()
        print('The server has stopped!')


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
            # TODO: Check if working queue has batches that run out of time.
            # If working queue is empty, do nothing.
            # If a batch is expired, pop it from the working queue and add it to the free queue.

    def _on_get_attack(self):
        # TODO: Send the current attack object. (It includes the current model object as well.)
        pass

    def _on_post_attack(self):
        # TODO: Override the current attack object with the recievd one. (It includes the new model object as well.)
        # Check if attack has "model" attribute and "perturb" method!
        pass

    def _on_get_clean_batch(self):
        # TODO: Pop a clean batch from the free queue, move it to the working queue and send the batch id and the batch itself.
        pass

    def _on_get_adv_batch(self):
        # TODO: Pop a batch from the done queue and send it.
        pass

    def _on_post_adv_batch(self):
        # TODO: Move the recieved batch from the woring queue to the done queue (using the also recieved batch id).
        # If time expired maximum patiente time, drop recieved batch and move it ti the free queue.
        # If done queue reached soft limit, drop recieved batch and move it to the free queue.
        # If the recieved batch is not in the working queue, drop it and do nothing.
        pass

    def _on_get_model_id(self):
        # TODO: Send the latest model id.
        pass

    def _on_get_model_state(self):
        # TODO: Send the state_dict of the latest model.
        pass

    def _on_post_model_state(self):
        # TODO: Update the current model with the recieved new model state_dict.
        pass

    def _on_post_dataset(self):
        # TODO: Construct the dataset object using the recieved class and arguments.
        pass

    def _on_post_dataloader(self):
        # TODO: Construct the dataloader object using the dataset object and the recieved arguments.
        pass


if __name__ == '__main__':
    # TODO: Parse arguments and start the server.
    Server('8080').run()

