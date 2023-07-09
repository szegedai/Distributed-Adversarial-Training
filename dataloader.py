import requests
import dill
import time


class DistributedAdversarialDataLoader:
    # TODO: Added proper error messages when the setup process is incorrect!
    def __init__(self, host, autoupdate_model=True, num_preprocessed_batches=10, max_batch_wait_time=300):
        self._host = host
        self._autoupdate_model = autoupdate_model
        self._queue_soft_limit = num_preprocessed_batches
        self._max_patiente = max_batch_wait_time
        self._attack = None
        self._num_batches = None
        self._next_batch_idx = 0

    def update_dataset(self, dataset_class, *dataset_args, **dataset_kwargs):
        self._send_data(
            f'http://{self._host}/dataset',
            [
                dataset_class,
                dataset_args,
                dataset_kwargs
            ]
        )

    def update_dataloader(self, *dataloader_args, **dataloader_kwargs):
        self._send_data(
            f'http://{self._host}/dataloader',
            [
                dataloader_args,
                dataloader_kwargs,
                self._max_patiente,
                self._queue_soft_limit
            ]
        )

    def update_attack(self, attack):
        assert hasattr(attack, 'model'), 'The attack object must have an attribute named "model"!'
        assert hasattr(attack, 'perturb') and callable(attack.perturb), 'The attack object must have a method named "perturb"!'

        self._send_data(f'http://{self._host}/attack', attack)
        self._attack = attack

    def update_model_state(self, model_state):
        self._send_data(f'http://{self._host}/model_state', model_state)

    def __iter__(self):
        num_batches = self._get_data(f'http://{self._host}/num_batches', max_retrys=5)
        self._num_batches = num_batches
        self._next_batch_idx = 0
        return self

    def __next__(self):
        if not self._num_batches or self._num_batches < self._next_batch_idx:
            raise Exception('Can not call "next(...)" before "iter(...)"!')
        if self._num_batches == self._next_batch_idx:
            raise StopIteration()

        batch = self._get_data(f'http://{self._host}/adv_batch', max_retrys=5)
        self._next_batch_idx += 1

        if self._autoupdate_model:
            self.update_model_state(self._attack.model.state_dict())

        return batch

    @staticmethod
    def _get_data(uri, max_retrys=-1):
        retry_count = 0
        while retry_count != max_retrys:
            response = requests.get(uri, verify=False)
            if response.status_code == 200:
                return dill.loads(response.content)
            time.sleep(1)
            retry_count += 1
        raise TimeoutError('Reached the maximum number of retrys while requesting data.')

    @staticmethod
    def _send_data(uri, data, max_retrys=-1):
        byte_data = dill.dumps(data)
        retry_count = 0
        while retry_count != max_retrys:
            response = requests.post(uri, data=byte_data, verify=False)
            if response.status_code == 200:
                return
            time.sleep(1)
            retry_count += 1
        raise TimeoutError('Reached the maximum number of retrys while sending data.')

