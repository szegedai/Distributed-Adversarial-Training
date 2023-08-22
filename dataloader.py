import requests
import dill
import torch
import io
import time

class DistributedAdversarialDataLoader:
    # TODO: Added proper error messages when the setup process is incorrect!
    # TODO: Handle the case when the connection to the server is lost temporarily and the session closes!
    def __init__(self, host, autoupdate_model=True, num_preprocessed_batches=10, max_patiente=20):
        self._host = host
        self._autoupdate_model = autoupdate_model
        self._queue_soft_limit = num_preprocessed_batches
        self._max_patiente = max_patiente
        self._model = None
        self._num_batches = None
        self._next_batch_idx = 0
        self._session = requests.Session()

    def update_dataset(self, dataset_class, *dataset_args, **dataset_kwargs):
        self._send_data(
            f'http://{self._host}/dataset',
            dill.dumps((dataset_class, dataset_args, dataset_kwargs)),
            self._session
        )

    def update_dataloader(self, *dataloader_args, **dataloader_kwargs):
        self._send_data(
            f'http://{self._host}/dataloader',
            dill.dumps((dataloader_args, dataloader_kwargs, self._max_patiente, self._queue_soft_limit)),
            self._session
        )

    def update_attack(self, attack_class, *attack_args, **attack_kwargs):
        attack_bytes = io.BytesIO()
        torch.save((attack_class, attack_args, attack_kwargs), attack_bytes, dill)
        self._send_data(
            f'http://{self._host}/attack', 
            attack_bytes.getvalue(),
            self._session
        )

    def update_model(self, model, new_architecture=False):
        model_bytes = io.BytesIO()
        torch.save(model, model_bytes, dill)
        self._send_data(
            f'http://{self._host}/model', 
            b''.join((
                new_architecture.to_bytes(1, 'big'),
                model_bytes.getvalue()
            )),
            self._session
        )
        self._model = model

    def __iter__(self):
        self._num_batches = int.from_bytes(
            self._get_data(f'http://{self._host}/num_batches', self._session, max_retrys=5),
            'big'
        )
        self._next_batch_idx = 0
        return self

    def __next__(self):
        if not self._num_batches or self._num_batches < self._next_batch_idx:
            raise Exception('Can not call "next(...)" before "iter(...)"!')
        if self._num_batches == self._next_batch_idx:
            raise StopIteration()

        batch = dill.loads(self._get_data(f'http://{self._host}/adv_batch', self._session))
        self._next_batch_idx += 1

        if self._autoupdate_model:
            self.update_model(self._model)

        return batch

    def __len__(self):
        if not self._num_batches:
            self._num_batches = int.from_bytes(
                self._get_data(f'http://{self._host}/num_batches', self._session, max_retrys=5),
                'big'
            )
        return self._num_batches

    def __del__(self):
        self._session.close()

    @staticmethod
    def _get_data(url, session, max_retrys=-1):
        retry_count = 0
        while retry_count != max_retrys:
            try:
                response = session.get(url, verify=False)
                if response.status_code == 200:
                    return response.content
            except (TimeoutError, ConnectionError) as e:
                raise Warning('The following error was raised during a GET request:\n' + str(e))
            time.sleep(1)
            retry_count += 1
        raise TimeoutError('Reached the maximum number of retrys while requesting data.')

    @staticmethod
    def _send_data(url, data, session, max_retrys=-1):
        retry_count = 0
        while retry_count != max_retrys:
            try:
                response = session.post(url, data=data, verify=False)
                if response.status_code == 200:
                    return
            except (TimeoutError, ConnectionError) as e:
                raise Warning('The following error was raised during a POST request:\n' + str(e))
            time.sleep(1)
            retry_count += 1
        raise TimeoutError('Reached the maximum number of retrys while sending data.')
    
