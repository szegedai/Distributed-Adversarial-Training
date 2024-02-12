import torch
import dill
import io
import traceback


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

@try_exc
def set_device(new_device):
    global device

    device = torch.device(new_device)

@try_exc
def perturb(encoded_data):
    global attack, model, device

    x, y = torch.load(io.BytesIO(encoded_data[8:]), device, dill)
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    new_x = attack.perturb(x, y)

    new_data = io.BytesIO()
    torch.save((new_x, y), new_data, dill, 5)

    return b''.join([
        encoded_data[:8].tobytes(),
        new_data.getvalue()
    ])

@try_exc
def update_attack(encoded_data):
    global attack, model, device

    attack_class, attack_args, attack_kwargs = torch.load(io.BytesIO(encoded_data.tobytes()), device, dill)
    encoded_data.release()
    attack = attack_class(model, *attack_args, **attack_kwargs)

@try_exc
def update_model(encoded_data):
    global attack, model, device

    model_class, model_args, model_kwargs = torch.load(io.BytesIO(encoded_data.tobytes()), device, dill)
    encoded_data.release()
    model = model_class(*model_args, **model_kwargs).to(device)

    if attack:
        attack.model = model

@try_exc
def update_model_state(encoded_data):
    global model, device

    new_state = torch.load(io.BytesIO(encoded_data.tobytes()), device, dill)
    model.load_state_dict(new_state)

