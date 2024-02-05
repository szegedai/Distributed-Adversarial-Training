import torch
import torch.utils.data as data
import time
import requests
import dill
import io
import torchvision
import torch.multiprocessing as mp
import queue
from copy import deepcopy

class DistributedAdversarialDataLoader(data.DataLoader):
    def __init__(
        self, 
        host='http://127.0.0.1:8080',
        batch_scale=1,
        num_workers=2,
        buffer_size=5,
        pin_memory_device='cpu'
    ):
        assert \
            (batch_scale >= 1 and not batch_scale % 1) or (batch_scale < 1 and batch_scale > 0), \
            'batch_scale must be a positive integer or a float between 1 and 0.'

        self.host = host
        self.pin_memory_device = pin_memory_device
        self._mp_ctx = mp.get_context('spawn')
        self._num_batches = -1
        self._num_processed_batches = 0
        self._batch_scale = batch_scale
        self._num_workers = num_workers
        self._session = requests.Session()
        self._batch_downloader_processes = []
        self._batch_queue = self._mp_ctx.Queue(buffer_size)
        self._model_uploader_process = None
        self._model_queue = self._mp_ctx.Queue()
        self._running = self._mp_ctx.Value('b', False, lock=True)

        dill.settings['recurse'] = True 

    def _get_data(self, to, timeout=None):
        response = self._session.get(f'{self.host}/{to}', verify=False, timeout=timeout)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception('GET request failed with status code', response.status_code)

    def _send_data(self, to, data, timeout=None):
        response = self._session.post(f'{self.host}/{to}', data=data, verify=False, timeout=timeout)
        if response.status_code == 200:
            return
        else:
            raise Exception('POST request failed with status code', response.status_code)

    def __len__(self):
        if self._num_batches < 0:
            try:
                self._num_batches = int.from_bytes(
                    self._get_data('num_batches', 5), 
                    'big', 
                    signed=False
                )
            except requests.exceptions.Timeout:
                raise Exception(
                    'Request to get the number of batches from server, timed out. This could happen if the remote server setup was not completed properly.'
                )
            if self._batch_scale > 1:
                # Integer division that rounds up, instead of down.
                self._num_batches = int(-(-self._num_batches // self._batch_scale))
            else:
                self._num_batches = int(self._num_batches // self._batch_scale)

        return self._num_batches

    def __iter__(self):
        if not self._running.value:
            self.start()

        return self

    def __next__(self):
        if self._num_processed_batches >= self._num_batches:
            self._num_processed_batches = 0
            raise StopIteration 

        self._num_processed_batches += 1
        return self._batch_queue.get(block=True, timeout=None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model_uploader_process'] = None
        state['_batch_downloader_processes'] = None
        return state

    def start(self):
        self._running.value = True

        self._model_uploader_process = self._mp_ctx.Process(target=self._model_uploader)
        self._model_uploader_process.start()

        len(self)  # Update num_batches if nessesary. 

        self._batch_downloader_processes.clear()
        for _ in range(self._num_workers):
            worker_process = self._mp_ctx.Process(target=self._batch_downloader)
            worker_process.start()
            self._batch_downloader_processes.append(worker_process)

    def stop(self):
        self._running.value = False

        # TODO: Make the cleanup proces better. Clear the queues first!

        self._model_uploader_process.terminate()
        self._model_uploader_process.join()
                
        for p in self._batch_downloader_processes:
            p.terminate()
            p.join()

    def update_attack(self, attack_class, *attack_args, **attack_kwargs):
        with io.BytesIO() as attack_bytes:
            torch.save((attack_class, attack_args, attack_kwargs), attack_bytes, dill)
            self._send_data('attack', attack_bytes.getvalue())
    
    def update_model(self, model):
        #self._model_queue.put(deepcopy(model).cpu())
        self._model_queue.put_nowait(model)

    def update_data(self, dataset_class, dataset_args, dataset_kwargs, dataloader_args, dataloader_kwargs):
        self._send_data(
            'data',
            dill.dumps((
                    dataset_class,
                    dataset_args,
                    dataset_kwargs, 
                    dataloader_args,
                    dataloader_kwargs
            ))
        )

    def set_parameters(self, max_patiente, queue_limit):
        self._send_data(
            'parameters',
            b''.join((
                max_patiente.to_bytes(8, 'big'),
                queue_limit.to_bytes(8, 'big'),
            )) 
        )

    def reset_server(self):
        self._send_data('reset', b'')

    def get_batch(self):
        with io.BytesIO(self._get_data('adv_batch')) as batch_bytes:
            batch = torch.load(
                    batch_bytes,
                    self.pin_memory_device,
                    dill
            )
        return batch

    def _batch_downloader(self):
        while self._running.value:
            if self._batch_scale < 1:
                original_batch = self.get_batch()
                batch_size = original_batch[0].size(0)
                indices = torch.arange(0, (1 + self._batch_scale) * batch_size, self._batch_scale * batch_size).to(dtype=torch.int64)
                for i in range(len(indices) - 1):
                    batch = (
                        original_batch[0][indices[i]:indices[i + 1]],
                        original_batch[1][indices[i]:indices[i + 1]]
                    )
                    self._batch_queue.put(batch, block=True, timeout=None)
            elif self._batch_scale == 1:
                self._batch_queue.put(self.get_batch(), block=True, timeout=None)
            else:
                xes, ys = [], []
                for _ in range(self._batch_scale):
                    x, y = self.get_batch()
                    xes.append(x)
                    ys.append(y)
                batch = (torch.cat(xes), torch.cat(ys))

                self._batch_queue.put(batch, block=True, timeout=None)

    def _model_uploader(self):
        while self._running.value:
            # Wait until there is at least one new model in the queue.
            model = self._model_queue.get(block=True, timeout=None)
            # If there are multiple models in thr queue, empty it and use the last one.
            try:
                while True:        
                    model = self._model_queue.get_nowait()
            except queue.Empty:
                pass

            model = deepcopy(model).cpu()

            with io.BytesIO() as model_bytes:
                torch.save(model, model_bytes, dill)
                self._send_data('model', model_bytes.getvalue())

