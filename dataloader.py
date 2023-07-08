import requests
import dill


class DistributedAdversarialDataLoader:
    def __init__(self, host, num_preprocessed_batches=10, max_batch_wait_time=300):
        self._host = host
        self._queue_soft_limit = num_preprocessed_batches
        self._max_patiente = max_batch_wait_time
        self._num_batches = None
        self._next_batch_idx = 0

    def update_dataset(self, dataset_class, *dataset_args, **dataset_kwargs):
        requests.post(
            f'http://{self._host}/dataset', 
            data=dill.dumps([
                dataset_class,
                dataset_args,
                dataset_kwargs
            ]),
            verify=False
        )

    def update_dataloader(self, dataloader_class, *dataloader_args, **dataloader_kwargs):
        requests.post(
            f'http://{self._host}/dataloader',
            data=dill.dumps([
                dataloader_class,
                dataloader_args,
                dataloader_kwargs,
                self._max_patiente,
                self._queue_soft_limit
            ]),
            verify=False
        )

    def update_attack(self, attack):
        requests.post(
            f'http://{self._host}/attack',
            data=dill.dumps(attack),
            verify=False
        )

    def update_model_state(self, model_state):
        requests.post(
            f'http://{self._host}/model_state',
            data=dill.dumps(model_state),
            verify=False
        )

    def __iter__(self):
        num_batches = dill.loads(requests.get(f'http://{self._host}/num_batches').content)
        if num_batches <= 0:
            raise Exception('Can not iterate dataloader on server! (The dataset or the dataloader was probably not initialised.)')
        self._num_batches = num_batches
        self._next_batch_idx = 0
        return self

    def __next__(self):
        if not self._num_batches or self._num_batches < self._next_batch_idx:
            raise Exception('Can not call "next(...)" before "iter(...)"!')
        if self._num_batches == self._next_batch_idx:
            raise StopIteration()

        batch = dill.loads(requests.get(f'http://{self._host}/adv_batch', verify=False).content)
        self._next_batch_idx += 1
        return batch

