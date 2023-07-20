import torch
import dill
import argparse
import json
import re
import time
import requests

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
        self._attack_id = None
        self._model_id = None

    def run(self):
        # TODO: Handle the case when the connection to the server is lost temporarily and the session closes!
        with requests.Session() as session:
            while True:
                # Update the model and/or attack if a newer one is available.
                response_bytes = self._get_data(f'http://{self.host}/ids', session)
                latest_attack_id, latest_mode_id = int.from_bytes(response_bytes[:8], 'big'), int.from_bytes(response_bytes[8:], 'big')
                if latest_attack_id != self._attack_id:
                    response_bytes = self._get_data(f'http://{self.host}/attack', session) 
                    attack_class, attack_args, attack_kwargs = dill.loads(response_bytes)
                    for arg in attack_args:
                        if isinstance(arg, torch.nn.Module):
                            arg.to(self.device)
                    self._attack = attack_class(*attack_args, **attack_kwargs)
                    self._attack_id = latest_attack_id
                if latest_mode_id != self._model_id:
                    response_bytes = self._get_data(f'http://{self.host}/model', session)
                    self._attack.model = dill.loads(response_bytes)
                    self._attack.model.to(self.device)
                    self._model_id = latest_mode_id

                # Request clean batch, perturb it and send back the result.
                response_bytes = self._get_data(f'http://{self.host}/clean_batch', session)
                batch_id, (x, y) = int.from_bytes(response_bytes[:8], 'big'), dill.loads(response_bytes[8:])
                x = x.to(self.device)
                y = y.to(self.device)

                self._send_data(
                    f'http://{self.host}/adv_batch', 
                    b''.join((
                        batch_id.to_bytes(8, 'big'),
                        dill.dumps((self._attack.perturb(x, y).cpu(), y.cpu()))
                    )), 
                    session
                )
 
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


if __name__ == '__main__':
    # Parse arguments, initialise a Node object and start it.
    parser = argparse.ArgumentParser(
        prog='Node',
        description='Worker node for distributed adversarial training.'
    )
    parser.add_argument(
        'host', 
        nargs='?', 
        default='127.0.0.1:8080', 
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

