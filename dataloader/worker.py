import torch
import torch.utils.data as data
import time
import json
import requests
import cloudpickle
import io
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
        pin_memory_device='cpu',
        store_extra_data=False
    ):
        assert \
            (batch_scale >= 1 and not batch_scale % 1) or (batch_scale < 1 and batch_scale > 0), \
            'batch_scale must be a positive integer or a float between 1 and 0.'

        self.host = host
        self.pin_memory_device = pin_memory_device
        self.store_extra_data = store_extra_data
        self._mp_ctx = mp.get_context('spawn')
        self._num_batches = 0
        self._num_processed_batches = 0
        self._batch_scale = batch_scale
        self._num_workers = num_workers
        self._session = requests.Session()
        self._batch_downloader_processes = []
        self._batch_queue = self._mp_ctx.Queue(buffer_size)
        self._model_uploader_process = None
        self._model_state_queue = self._mp_ctx.Queue()
        self._extra_data_queue = self._mp_ctx.Queue()
        self._running = self._mp_ctx.Value('b', False, lock=True)
        self._required_to_start = {
            'update_attack': True,
            'update_dataset': True,
            'update_dataloader': True,
            'update_model': True,
            'update_model_state': True,
            'set_parameters': True
        }

        #dill.settings['recurse'] = True

    @staticmethod
    def sync_external_modules(modules):
        for m in modules:
            cloudpickle.register_pickle_by_value(m)

    @staticmethod
    def unsync_external_modules(modules):
        for m in modules:
            cloudpickle.unregister_pickle_by_value(m)

    @staticmethod
    def _serialize_data(obj):
        with io.BytesIO() as data_bytes:
            torch.save(obj, data_bytes)
            return data_bytes.getvalue()

    @staticmethod
    def _deserialize_data(data, device=None):
        with io.BytesIO(data) as data_bytes:
            return torch.load(data_bytes, device)

    def _get_data(self, to, timeout=None):
        response = self._session.get(f'{self.host}/{to}', verify=False, timeout=timeout)
        if response.status_code == 200:
            return response.content, response.headers.get('X-Extra-Data', '{}')
        else:
            raise Exception('GET request failed with status code', response.status_code)

    def _send_data(self, to, data, timeout=None):
        response = self._session.post(f'{self.host}/{to}', data=data, verify=False, timeout=timeout)
        if response.status_code == 200:
            return
        else:
            raise Exception('POST request failed with status code', response.status_code)

    def __len__(self):
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
        for method_name, not_called in self._required_to_start.items():
            if not_called:
                raise Exception(f'It is required to call "{method_name}" in order to start the dataloader procedure!')

        self._running.value = True

        self._model_uploader_process = self._mp_ctx.Process(target=self._model_uploader)
        self._model_uploader_process.start()

        # Setup the correct number of batches.
        self._num_batches = int.from_bytes(
            self._get_data('num_batches')[0], 
            'big', 
            signed=False
        )
        if self._batch_scale > 1:
            # Integer division that rounds up, instead of down.
            self._num_batches = int(-(-self._num_batches // self._batch_scale))
        else:
            self._num_batches = int(self._num_batches // self._batch_scale)

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
        self._send_data(
            'attack', 
            cloudpickle.dumps(
                (attack_class, attack_args, attack_kwargs),
                protocol=5
            )
        )
        self._required_to_start['update_attack'] = False
    
    def update_dataset(self, dataset_class, *dataset_args, **dataset_kwargs):
        self._send_data(
            'dataset', 
            cloudpickle.dumps(
                (dataset_class, dataset_args, dataset_kwargs),
                protocol=5
            )
        )
        self._required_to_start['update_dataset'] = False

    def update_dataloader(self, dataloader_class, *dataloader_args, **dataloader_kwargs):
        self._send_data(
            'dataloader',
            cloudpickle.dumps(
                (dataloader_class, dataloader_args, dataloader_kwargs),
                protocol=5
            )
        )
        self._required_to_start['update_dataloader'] = False

    def update_model(self, model_class, *model_args, **model_kwargs):
        self._send_data(
            'model',
            cloudpickle.dumps(
                (model_class, model_args, model_kwargs),
                protocol=5
            )
        )
        self._required_to_start['update_model'] = False

    def update_model_state(self, model_state):
        self._model_state_queue.put_nowait(model_state)
        self._required_to_start['update_model_state'] = False

    def set_parameters(self, max_patiente, queue_limit):
        self._send_data(
            'parameters',
            b''.join((
                max_patiente.to_bytes(8, 'big'),
                queue_limit.to_bytes(8, 'big'),
            )) 
        )
        self._required_to_start['set_parameters'] = False

    def reset_server(self):
        self._send_data('reset', b'')

    def get_batch(self):
        data, extra_data = self._get_data('adv_batch')
        return self._deserialize_data(
            data, 
            self.pin_memory_device
        ), json.loads(extra_data)

    def get_extra_data(self, block=True, timeout=None):
        return self._extra_data_queue.get(block, timeout)

    def _batch_downloader(self):
        while self._running.value:
            if self._batch_scale == 1:
                batch, extra_data = self.get_batch()
                self._batch_queue.put(batch, block=True, timeout=None)
                if self.store_extra_data:
                    self._extra_data_queue.put_nowait(extra_data)
            elif self._batch_scale < 1:
                original_batch, extra_data = self.get_batch()
                batch_size = original_batch[0].size(0)
                indices = torch.arange(0, (1 + self._batch_scale) * batch_size, self._batch_scale * batch_size).to(dtype=torch.int64)
                for i in range(len(indices) - 1):
                    batch = (
                        original_batch[0][indices[i]:indices[i + 1]],
                        original_batch[1][indices[i]:indices[i + 1]]
                    )
                    self._batch_queue.put(batch, block=True, timeout=None)
                if self.store_extra_data:
                    self._extra_data_queue.put_nowait(extra_data)
            else:
                xes, ys = [], []
                extra_datas = []
                for _ in range(self._batch_scale):
                    (x, y), extra_data = self.get_batch()
                    xes.append(x)
                    ys.append(y)
                    extra_datas.append(extra_data)
                batch = (torch.cat(xes), torch.cat(ys))

                self._batch_queue.put(batch, block=True, timeout=None)
                if self.store_extra_data:
                    for extra_data in extra_datas:
                        self._extra_data_queue.put_nowait(extra_data)

    def _model_uploader(self):
        while self._running.value:
            # Wait until there is at least one new model in the queue.
            model_state = self._model_state_queue.get(block=True, timeout=None)
            # If there are multiple models in thr queue, empty it and use the last one.
            try:
                while True:        
                    model_state = self._model_state_queue.get_nowait()
            except queue.Empty:
                pass

            self._send_data(
                'model_state',
                self._serialize_data(deepcopy(model_state))
            )

