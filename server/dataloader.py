import traceback
import torch
import pickle
import io

def try_exc(f):
    def g(*args):
        try:
            return f(*args)
        except:
            traceback.print_exc()
            exit(-1)
    return g

@try_exc
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
def update_dataset(encoded_data):
    global dataset

    ds_class, ds_args, ds_kwargs = pickle.loads(encoded_data.tobytes())
    dataset = ds_class(*ds_args, **ds_kwargs)

@try_exc
def update_dataloader(encoded_data):
    global dataset, dataloader, dataloader_iter

    dl_class, dl_args, dl_kwargs = pickle.loads(encoded_data.tobytes())
    dataloader = dl_class(dataset, *dl_args, **dl_kwargs)
    dataloader_iter = iter(dataloader)

@try_exc
def get_num_batches():
    global dataloader

    return len(dataloader)

@try_exc
def get_clean_batch():
    global dataloader, dataloader_iter

    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)

    batch_bytes = io.BytesIO()
    torch.save(batch, batch_bytes)

    return batch_bytes.getvalue()

