import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
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
        num_workers=2,
        buffer_size=5,
        pin_memory_device='cpu'
    ):
        self.host = host
        self.pin_memory_device = pin_memory_device
        self._mp_ctx = mp.get_context('spawn')
        self._model = None
        self._num_batches = -1
        self._num_processed_batches = 0
        self._num_workers = num_workers
        self._session = requests.Session()
        self._batch_downloader_processes = []
        self._batch_queue = self._mp_ctx.Queue(buffer_size)
        self._model_uploader_process = None
        self._model_queue = self._mp_ctx.Queue(1)
        self._running = self._mp_ctx.Value('b', False, lock=True)

        dill.settings['recurse'] = True 

    def _get_data(self, to):
        response = self._session.get(f'{self.host}/{to}', verify=False)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception('GET request failed with status code', response.status_code)

    def _send_data(self, to, data):
        response = self._session.post(f'{self.host}/{to}', data=data, verify=False)
        if response.status_code == 200:
            return
        else:
            raise Exception('POST request failed with status code', response.status_code)

    def __len__(self):
        if self._num_batches < 0:
            self._num_batches = int.from_bytes(
                self._get_data('num_batches'), 
                'big', 
                signed=False
            )

        return self._num_batches

    def __iter__(self):
        if not self._running.value:
            self.start()

        return self

    def __next__(self):
        if self._num_processed_batches >= self._num_batches:
            self._num_processed_batches = 0
            raise StopIteration

        batch = self._batch_queue.get(block=True, timeout=None)
        self._num_processed_batches += 1
        return batch

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model_uploader_process'] = None
        state['_batch_downloader_processes'] = None
        return state

    def start(self):
        self._running.value = True

        self._model_uploader_process = self._mp_ctx.Process(target=self._model_uploader)
        self._model_uploader_process.start()

        len(self)  # Update num_batches is nessesary. 

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
        self._model = model

        # Empty the model queue if there is a congestion, and put the lates model in.
        try:
            self._model_queue.get_nowait()
        except queue.Empty:
            self._model_queue.put_nowait(deepcopy(model).cpu())

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
            self._batch_queue.put(self.get_batch(), block=True, timeout=None)

    def _model_uploader(self):
        while self._running.value:
            model = self._model_queue.get(block=True, timeout=None)

            with io.BytesIO() as model_bytes:
                torch.save(model, model_bytes, dill)
                self._send_data('model', model_bytes.getvalue())


class LinfPGDAttack:
    def __init__(self, model, loss_fn, eps, step_size, num_steps, random_start=True, bounds=(0.0, 1.0)):
        self.model = model
        self.loss_fn = loss_fn

        self.eps = eps
        self.step_size = step_size
        self.num_steps = num_steps
        self.bounds = bounds

        self.random_start = random_start

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @torch.no_grad()
    def perturb(self, x, y):
        delta = torch.zeros_like(x, dtype=self.dtype, device=self.device)
        if self.random_start:
            delta = delta.uniform_(-self.eps, self.eps)
            delta = (x + delta).clamp(*self.bounds) - x

        for _ in range(self.num_steps):
            with torch.enable_grad():
                delta.requires_grad = True
                loss = self.loss_fn(self.model(x + delta), y)
                grads = torch.autograd.grad(loss, delta)[0]
            delta = delta + self.step_size * torch.sign(grads)
            delta = delta.clamp(-self.eps, self.eps)
            delta = (x + delta).clamp(*self.bounds) - x
        return x + delta

def main():
    data_path = '../cifar_data/cifar10'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    train_loader = DistributedAdversarialDataLoader(pin_memory_device=device)

    net = torchvision.models.resnet18(num_classes=10).to(device)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip()
    ])

    train_loader.update_model(net)

    train_loader.update_attack(
        LinfPGDAttack,
        net,
        torch.nn.CrossEntropyLoss(),
        8 / 255,
        2 / 255,
        10
    )

    train_loader.set_parameters(max_patiente=100000, queue_limit=5)
    train_loader.update_data(
        torchvision.datasets.CIFAR10, 
        [
            data_path
        ], 
        {
            "train": True, 
            "transform": train_transform, 
            "download": True
        },
        [],
        {
            'batch_size': 256, 
            'shuffle': True, 
            'num_workers': 4, 
            'multiprocessing_context': 'spawn', 
            'persistent_workers': True
        }
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    running_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total = labels.size(0)
            correct = (predicted == labels).sum().item()

            if(i % 2 == 0):
                train_loader.update_model(net)

            if i % 1 == 0:    # Print average loss and accuracy every 200 mini-batches
                print(f"[Epoch {epoch+1}, Batch {i+1} loss: {running_loss}, accuracy: {100 * correct / total}%")
                running_loss = 0.0
                correct = 0
                total = 0

    train_loader.stop()
    print("Training finished.")

if __name__ == '__main__':
    main()

