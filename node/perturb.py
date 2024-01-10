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
    print("Py: set_device - start")
    global device
    device = torch.device(new_device)
    print("Py: set_device - end")

@try_exc
def perturb(encoded_data):
    global attack, device

    x, y = torch.load(io.BytesIO(encoded_data[8:]), device, dill)
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    new_x = attack.perturb(x, y)

    new_data = io.BytesIO()
    torch.save((new_x, y), new_data, dill)

    #encoded_data[8:] = new_data.getvalue

    return b''.join([
        encoded_data[:8].tobytes(),
        new_data.getvalue()
    ])

@try_exc
def update_attack(encoded_data):
    print("Py: update_attack - start")
    global attack, device
    print_bytes(encoded_data.tobytes())
    attack_class, attack_args, attack_kwargs = torch.load(io.BytesIO(encoded_data.tobytes()), device, dill)
    encoded_data.release()
    for arg in attack_args:
        if isinstance(arg, torch.nn.Module):
            arg.to(device)
    attack = attack_class(*attack_args, **attack_kwargs)
    print("Py: update_attack - end")

@try_exc
def update_model(encoded_data):
    print("Py: update_model - start")
    global attack, device
    print_bytes(encoded_data.tobytes())
    new_model = torch.load(io.BytesIO(encoded_data.tobytes()), device, dill)
    print(new_model)
    encoded_data.release()
    attack.model = new_model
    attack.model.to(device)
    print("Py: update_mode - end")

