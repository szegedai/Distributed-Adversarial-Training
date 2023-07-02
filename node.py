import torch
import pickle
import requests
import argparse
import json
import re


class Node:
    # Consider adding support for HTTPS too!
    # TODO: Manage the scenario when the node is not able to access the server temporarily.
    def __init__(self, host, device):
        # Validate device string.
        assert re.compile(
            r'^(cuda(:[0-9]+)?|mps|[ctx]pu)$'
        ).search(device), f'"{device}" is not a valid device!'

        # Validate if host address is a valid IPv4 or domain name.
        # TODO: Add support for IPv6!
        assert re.compile(
            r'^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])(:[1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?$'
        ).search(host) or re.compile(
            r'^((((2[0-5]{0,2}|1[0-9]{0,2}|[1-9][0-9]?|0)\.){3}(2[0-5]{0,2}|1[0-9]{0,2}|[1-9][0-9]?)|localhost)(:[1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?)$'
        ).search(host), f'"{host}" is not a valid IPv4 address or domain name!'

        self.host = host
        self.device = torch.device(device)
        self._attack = None
        self._model = None
        self._model_id = None

    def run(self):
        # Request and setup the attack and model objects.
        self._attack, self._model = self._get_data(f'http://{self.host}/init')
        self._model.to(self.device)
        self._attack.model = self._model

        # Request clean batches and perturb them, until the program shuts down.
        while True:
            clean_batch = self._get_data(f'http://{self.host}/clean_batch')
            self._send_data(
                f'http://{self.host}/adv_batch', 
                self._attack.perturb(*clean_batch)
            )

            # Update the model state if a newer one is available.
            latest_mode_id = self._get_data(f'http://{self.host}/model_id')
            if latest_mode_id != self._model_id:
                self._model.load_state_dict(self._get_data(f'http://{self.host}/model_state'))
                self._model_id = latest_mode_id

    @staticmethod
    def _get_data(uri):
        return pickle.loads(requests.get(uri, verify=False).content)

    @staticmethod
    def _send_data(uri, data):
        requests.put(uri, pickle.dumps(data), verify=False)


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

    
    Node(**args).run()

