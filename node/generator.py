import torch
import io
import pickle
import traceback
import torch.multiprocessing as mp
from time import time


def try_exc(f):
    def g(*args):
        try:
            return f(*args)
        except:
            traceback.print_exc()
            exit(-1)
    return g

def print_bytes(data):
    print('Py:', end=' ')
    for i in range(10):
        if i < len(data):
            print(f'{data[i]:02X}', end=' ')
        else:
            break
    print('...', end=' ')
    
    for i in range(len(data) - 10, len(data)):
        if i >= 0:
            print(f'{data[i]:02X}', end=' ')
        else:
            break
    print()

device = None

attack_data = None
attack = None

model_data = None
model = None

mp_ctx = mp.get_context('spawn')
generator_running = mp_ctx.Value('b', False)
clean_batch_receiver, clean_batch_sender = mp_ctx.Pipe(duplex=False)
adv_batch_receiver, adv_batch_sender = mp_ctx.Pipe(duplex=False)
model_state_receiver, model_state_sender = mp_ctx.Pipe(duplex=False)
generator_process = None

@try_exc
def set_device(new_device):
    global device

    device = torch.device(new_device)

@try_exc
def update_attack(encoded_data):
    global generator_running, attack_data

    attack_data = encoded_data.tobytes()
    encoded_data.release()

    if generator_running.value:
        stop_generator()
    start_generator()

@try_exc
def update_model(encoded_data):
    global generator_process, model_data

    model_data = encoded_data.tobytes()
    encoded_data.release()

    if generator_running.value:
        stop_generator()
    start_generator()

@try_exc
def push_batch(encoded_data):
    global device, clean_batch_sender

    id_bytes = encoded_data[:8].tobytes()
    x, y = torch.load(io.BytesIO(encoded_data[8:]), device)
    clean_batch_sender.send((id_bytes, x, y))

@try_exc
def pop_batch():
    global adv_batch_receiver

    id_bytes, x, y = adv_batch_receiver.recv()
    with io.BytesIO() as encoded_data:
        torch.save((x, y), encoded_data)
        return b''.join([
            id_bytes,
            encoded_data.getvalue()
        ])

@try_exc
def push_model_state(encoded_data):
    global device, model_state_sender

    new_state = torch.load(io.BytesIO(encoded_data.tobytes()), device)
    model_state_sender.send(new_state)

@try_exc
def start_generator():
    global mp_ctx, device, model_data, attack_data, \
    generator_running, clean_batch_receiver, adv_batch_sender, model_state_receiver, generator_process

    if model_data and attack_data:
        generator_running.value = True
        generator_process = mp_ctx.Process(
            target=run_generator_loop, 
            args=(
                device, model_data, attack_data, 
                generator_running, clean_batch_receiver, adv_batch_sender, model_state_receiver
            )
        )
        generator_process.start()

@try_exc
def stop_generator():
    global generator_running, generator_process

    generator_running.value = False
    generator_process.join()

def run_generator_loop(device, model_data, attack_data, generator_running, clean_batch_receiver, adv_batch_sender, model_state_receiver):
    try:
        model_class, model_args, model_kwargs = pickle.loads(model_data)
        attack_class, attack_args, attack_kwargs = pickle.loads(attack_data)
        model = model_class(*model_args, **model_kwargs).to(device)
        attack = attack_class(model, *attack_args, **attack_kwargs)

        while generator_running:
            if model_state_receiver.poll():
                model.load_state_dict(model_state_receiver.recv())

            id_bytes, x, y = clean_batch_receiver.recv()

            x = attack.perturb(x, y)

            adv_batch_sender.send((id_bytes, x.clone(), y.clone()))
    except:
        traceback.print_exc()
        exit(-1)

