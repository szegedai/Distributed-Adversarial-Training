import torch
import requests
import argparse
import json
import re


class Node:
    def __init__(self, host, device):
        self.host = host
        self.device = torch.device(device)
        self._attack = None
        self._model = None
        self._model_id = None

    def run(self):
        # TODO: Start running the node, somewhat like this:
        # 1) request attack and model class
        # 2) construct model object
        # 3) update local model
        # 4) loop:
        # 5)     request clean batch
        # 6)     generate adversarial batch
        # 7)     send adversarial batch
        # 8)     update local model if needed
        pass

    def _request_init_data(self):
        # TODO: Make an HTTP GET request to the ES (Execution Server) to get the attack object and the model class.
        pass

    def _request_batch(self):
        # TODO: Make an HTTP GET request to the ES for a new batch.
        pass

    def _send_batch(self, batch):
        # TODO: Compress and send the perturbed batch to the ES via HTTP PUT.
        pass

    def _update_model(self):
        # TODO: Make an HTTP GET request to the ES for the latest model id and make an additional request for the
        # latest model state if the local model id does not equal to the recieved latest model id.
        pass


if __name__ == '__main__':
    # Parse arguments, initialise a Node object and start it.
    parser = argparse.ArgumentParser(
        prog='Node',
        description='Worker node for distributed adversarial training.'
    )
    parser.add_argument(
        'host', 
        nargs='?', 
        default='127.0.0.1', 
        help='The IP address of the Execution Server to connect.'
    )
    parser.add_argument(
        'device', 
        nargs='?', 
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        help='The local device to use for PyTorch operations.'
    )
    parser.add_argument(
        '-c', '--config', 
        default=None, 
        help='Config file to use instead of arguments.'
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config) as fp:
            args = json.load(fp)
        assert args['host'], 'The config file must contain the IP address of the Executions Server!'
        assert args['device'], 'The config file must contain the name of the local device!'
    else:
        args = vars(args)
        del args['config']

    # Validate device string.
    assert re.compile(
        r'^(cuda(:[0-9]+)?|mps|[ctx]pu)$'
    ).search(args['device']), f'"{args["device"]}" is not a valid device!'

    # Validate if host address is a valid IPv4 or domain name.
    assert re.compile(
        r'^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$'
    ).search(args['host']) or re.compile(
        r'^(((2[0-5]{0,2}|1[0-9]{0,2}|[1-9][0-9]?|0)\.){3}(2[0-5]{0,2}|1[0-9]{0,2}|[1-9][0-9]?)|localhost)$'
    ).search(args['host']), f'"{args["host"]}" is not a valid IP address or domain name!'

    Node(**args).run()

