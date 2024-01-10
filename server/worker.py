import torch
import torch.utils.data as data
import time
import requests
import dill
import io
import torchvision
import torch.nn as nn
import torch.optim as optim

from requests import get
from multiprocessing import Queue, Event, Process

class ContinuousHTTPWorkerDataLoader(data.DataLoader):
    def __init__(self, 
        host='127.0.0.1:8080',
        autoupdate_model=True,
        num_preprocessed_batches=10,
        max_patiente=20,
        num_workers=2,
        worker_queue_limit=15,
        pin_memory_device='cuda:0'
    ):
        self._host = host
        self._autoupdate_model = autoupdate_model
        self._queue_soft_limit = num_preprocessed_batches
        self._max_patiente = max_patiente
        self._model = None
        self._num_batches = None
        self._next_batch_idx = 0
        self._session = requests.Session()

        dill.settings['recurse'] = True

        self.num_workers = num_workers
        self.worker_queue_limit = worker_queue_limit
        self.pin_memory_device = pin_memory_device

        self.batch_queue = Queue()
        self.ready_event = Event()

        self.worker_processes = []
        for _ in range(self.num_workers):
            worker_process = Process(target=self.worker_process)
            worker_process.start()
            self.worker_processes.append(worker_process)

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

    def __iter__(self):
        self.ready_event.wait()

        while True:
            try:
                batch = self.batch_queue.get()
                yield batch

            except Exception as e:
                print('Failed to get batch:', e)

    def update_attack(self, attack_class, *attack_args, **attack_kwargs):
        attack_bytes = io.BytesIO()
        torch.save((attack_class, attack_args, attack_kwargs), attack_bytes, dill)
        self._send_data(
            f'http://{self._host}/attack', 
            attack_bytes.getvalue(),
            self._session
        )
    
    def update_model(self, model, new_architecture=False):
        self._model = model

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


    def update_data(self, dataset_class, dataset_args, dataset_kwargs, dataloader_args, dataloader_kwargs):
        self._send_data(
            f'http://{self._host}/data',
            dill.dumps(
                (
                    dataset_class,
                    dataset_args,
                    dataset_kwargs, 
                    dataloader_args,
                    dataloader_kwargs
                )
            ),
            self._session
        )

    def set_parameters(self, max_patiente, queue_soft_limit):
        self._send_data(
            f'http://{self._host}/parameters',
            b''.join((
                max_patiente.to_bytes(8, 'big'),
                queue_soft_limit.to_bytes(8, 'big'),
            )), 
            self._session
        )

    def reset_server(self):
        self._send_data(f'http://{self._host}/data', b'', self._session)

    def get_batch(self):
        batch = torch.load(
            io.BytesIO(
                self._get_data(f'http://{self._host}/adv_batch', self._session)),
                self.pin_memory_device,
                dill
        )
        self._next_batch_idx += 1

        return batch


    def worker_process(self):
        self.ready_event.set()

        while True:
            if(self.worker_queue_limit - self.num_workers < self.batch_queue.qsize()):
                time.sleep(0.001)
            else:

                try:
                    data = self.get_batch()

                    if self.pin_memory_device:
                        image = data[0].to(self.pin_memory_device, non_blocking=True)
                        label = data[1].to(self.pin_memory_device, non_blocking=True)
                    else:
                        image = data[0]
                        label = data[1]

                    self.batch_queue.put((image, label))

                    time.sleep(0.01)
                except Exception as e:
                    print('Worker failed:', e)


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

train_loader = ContinuousHTTPWorkerDataLoader()

data_path = '../cifar_data/cifar10'
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')

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

train_loader.set_parameters(max_patiente=100000, queue_soft_limit=5)
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
        "batch_size": 128, 
        "shuffle": True, 
        "num_workers": 3, 
        "multiprocessing_context": 'spawn', 
        "persistent_workers": True
    }
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

running_loss = 0.0
correct = 0
total = 0

for epoch in range(10):
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

print("Training finished.")
