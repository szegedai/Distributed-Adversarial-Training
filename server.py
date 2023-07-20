import torch
import dill
import argparse
import asyncio
from aiohttp import web
from heapq import heappush, heappop


class Server:
    # TODO:
    # 2) Handle the case when a request comes in before the dataset and dataloader objects are initialised!
    # 3) Handle the case when dataloader constructuion is called before dataset construction!
    def __init__(self, port):
        self._port = int(port)

        self._http_server = web.Application()

        self._attack_info = None
        self._latest_attack_id = -1
        self._model = None
        self._latest_model_id = -1

        self._dataset = None
        self._dataloader = None
        self._dataloader_iter = None
        self._queue_soft_limit = None

        self._batch_store = dict()
        self._old_batch_store = dict()  # stores the unperturbed batches in case the batches in the done queue has to be redone.

        self._free_q = list()  # Need priority so batches that need redistribution, get to the top.
        self._working_q = dict()  # Fast access when searching for specific batch id and iterable to check for timeouts (this needs a lock).
        self._done_q = list()  # Need priority so batches that were redistributed, get to the top.

        self._latest_unused_batch_id = 0
        self._max_patiente = None

    async def run(self):
        event_loop = asyncio.get_event_loop()
        event_loop.create_task(self._run_timeout_handler())
        await self._run_request_handler()

    async def _load_batch(self):
        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self._dataloader)
            batch = next(self._dataloader_iter)
        id = self._latest_unused_batch_id
        self._latest_unused_batch_id += 1
        self._batch_store[id] = dill.dumps(batch)
        heappush(self._free_q, id)

    async def _run_request_handler(self):
        self._http_server.add_routes([
            web.get('/attack', self._on_get_attack),
            web.post('/attack', self._on_post_attack),
            web.get('/clean_batch', self._on_get_clean_batch),
            web.get('/adv_batch', self._on_get_adv_batch),
            web.post('/adv_batch', self._on_post_adv_batch),
            web.get('/ids', self._on_get_ids),
            web.get('/model', self._on_get_model),
            web.post('/model', self._on_post_model),
            web.post('/dataset', self._on_post_dataset),
            web.post('/dataloader', self._on_post_dataloader),
            web.get('/num_batches', self._on_get_num_batches)
        ])

        runner = web.AppRunner(self._http_server, keepalive_timeout=600)
        await runner.setup()
        try:
            site = web.TCPSite(runner, '0.0.0.0', self._port)
            await site.start()
            # Keep the server running forever.
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()
        #web.run_app(self._http_server, host='0.0.0.0', port=self._port, keepalive_timeout=600)

    async def _run_timeout_handler(self):
        while True:
            await asyncio.sleep(2)
            # Check if working queue has batches that run out of time.
            # If a batch is expired, pop it from the working queue and add it to the free queue.
            for batch_id in list(self._working_q.keys()):
                if self._working_q[batch_id] - self._latest_model_id > self._max_patiente:
                    del self._working_q[batch_id]
                    heappush(self._free_q, batch_id)

    async def _on_get_attack(self, _):
        # Send the current attack class and arguments.
        return web.Response(body=self._attack_info)

    async def _on_post_attack(self, request):
        # Override the current attack class and args with the recieved one.
        self._attack_info = await request.content.read()
        self._latest_attack_id += 1

        # Move all batches to the free queue to resend them.
        for batch_id in self._working_q.keys():
            heappush(self._free_q, batch_id)
        for batch_id in self._done_q:
            heappush(self._free_q, batch_id)
        self._working_q.clear()
        self._done_q.clear()
        return web.Response()
            
    async def _on_get_clean_batch(self, _):
        # Pop a clean batch from the free queue, move it to the working queue and send the batch id and the batch itself.
        while not len(self._free_q):
            # There are no clean batches available, probably because the full initialisation process is not yet finished.
            # Wait and check again until a clean batch is available.
            await asyncio.sleep(0.5)
        batch_id = heappop(self._free_q)
        self._working_q[batch_id] = self._latest_model_id
        return web.Response(body=b''.join((
            batch_id.to_bytes(8, 'big'),
            self._batch_store[batch_id]
        )))

    async def _on_get_adv_batch(self, _):
        # Pop a batch id from the done queue, remove it from the batch store and send it.
        while not len(self._done_q):
            # There are no adversarial batches available.
            # This can happen if the initialisation process is not yet finished or 
            # if the nodes have not yet had enough time to generate new adversarial batches.
            # Wait and check again until an adversarial batch is available.
            await asyncio.sleep(0.5)
        batch_id = heappop(self._done_q)
        del self._old_batch_store[batch_id]
        return web.Response(body=self._batch_store.pop(batch_id))

    async def _on_post_adv_batch(self, request):
        # Move the recieved batch id from the woring queue to the done queue and override the batch in the batch store.
        # If done queue reached soft limit, drop recieved batch and move it to the free queue.
        # If the recieved batch is not in the working queue, drop it and do nothing.
        # If the batch has expired, remove it from the working queue and add it to the free queue.
        data = await request.content.read()
        batch_id, batch = int.from_bytes(data[:8], 'big'), data[8:]
        if not (batch_id in self._working_q):
            return web.Response()
        if self._working_q[batch_id] - self._latest_model_id > self._max_patiente or len(self._done_q) >= self._queue_soft_limit:
            del self._working_q[batch_id]
            heappush(self._free_q, batch_id)
            return web.Response()
        del self._working_q[batch_id]
        self._old_batch_store[batch_id] = self._batch_store[batch_id]
        self._batch_store[batch_id] = batch
        heappush(self._done_q, batch_id)
        await self._load_batch()
        return web.Response()

    async def _on_get_ids(self, _):
        # Send the latest attack and model ids.
        while self._latest_attack_id == -1 or self._latest_model_id == -1:
            # The server has not yet finished the initialisation process.
            await asyncio.sleep(0.5)
        return web.Response(body=b''.join((
            self._latest_attack_id.to_bytes(8, 'big'), 
            self._latest_model_id.to_bytes(8, 'big')
        )))

    async def _on_get_model(self, _):
        return web.Response(body=self._model)

    async def _on_post_model(self, request):
        data = await request.content.read()
        new_architecture, self._model = data[0], data[1:]
        self._latest_model_id += 1

        if new_architecture:
            # Move all batches to the free queue to resend them.
            for batch_id in self._working_q.keys():
                heappush(self._free_q, batch_id)
            for batch_id in self._done_q:
                heappush(self._free_q, batch_id)
            self._working_q.clear()
            self._done_q.clear()
        return web.Response()

    async def _on_post_dataset(self, request):
        # Clear all queues and dataloader. The user have to update the dataloader to be able to start working.
        self._free_q.clear()
        self._working_q.clear()
        self._done_q.clear()
        self._dataloader = None
        self._dataloader_iter = None

        # Construct the dataset object using the recieved class and arguments.
        dataset_class, args, kwargs = dill.loads(await request.content.read())
        self._dataset = dataset_class(*args, **kwargs) 
        return web.Response()

    async def _on_post_dataloader(self, request):
        # Construct the dataloader object using the dataset object and the recieved arguments.
        # Clear the batch store and all queues since a new dataloader was given so the ongoing and done batches are irrelevant.
        while not self._dataset:
            # There was no dataset to use for the dataloader.
            await asyncio.sleep(0.5)
        self._free_q.clear()
        self._working_q.clear()
        self._done_q.clear()
        self._batch_store.clear()

        args, kwargs, self._max_patiente, self._queue_soft_limit = dill.loads(await request.content.read())
        self._dataloader = torch.utils.data.DataLoader(self._dataset, *args, **kwargs)
        self._dataloader_iter = iter(self._dataloader)
        # Preload batches as fast as possible to not starve the nodes on startup.
        for _ in range(self._queue_soft_limit):
            await self._load_batch()
        return web.Response()

    async def _on_get_num_batches(self, _):
        while not self._dataloader:
            # There was no dataloader to use.
            await asyncio.sleep(0.5)
        return web.Response(body=len(self._dataloader).to_bytes(8, 'big'))


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
    
    asyncio.run(Server(**args).run())

