import traceback
import torch
import dill
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
def update_dataset(data_bytes):
    global dataloader, dataloader_iter, dataset

    dataloader = None
    dataloader_iter = None

    dataset_class, args, kwargs = dill.loads(data_bytes)
    dataset = dataset_class(*args, **kwargs)

@try_exc
def update_dataloader(data_bytes):
    global dataloader, dataloader_iter, dataset

    args, kwargs = dill.loads(data_bytes)
    dataloader = torch.utils.data.DataLoader(dataset, *args, **kwargs)
    dataloader_iter = iter(dataloader)

@try_exc
def get_num_batches():
    global dataloader

    return len(dataloader).to_bytes(8, 'big')

@try_exc
def get_clean_batch():
    global dataloader, dataloader_iter

    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)

    batch_bytes = io.BytesIO()
    torch.save(batch, batch_bytes, dill)
    return batch_bytes.getvalue()

