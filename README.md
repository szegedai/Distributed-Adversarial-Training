# Adversarial Training Node

This script represents a worker node for distributed adversarial training. It connects to an Execution Server to participate in the training process.

## Prerequisites

- Python 3.9 or later
- Docker (optional, for running the script as a Docker container)

## Getting Started

1. Clone the repository:
```bash
git clone <repository-url>
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run node with argument parameters:
```bash
python node.py <host> <device>
```
4. Run node with config file:
```bash
python node.py -c config.json
```

## Running as a Docker Container (optional)
1. Build the Docker image:
```bash
docker build -t adversarial-training-node .
```
2. Run a Docker container:
```bash
docker run -d --name node-container adversarial-training-node
```

This command starts the container in detached mode with the name node-container.
Note: Adjust the Dockerfile and commands as needed for your specific setup.

## Configuration
Instead of providing the host and device as command-line arguments, you can use a JSON configuration file (config.json) to specify these values. The file should have the following structure:
```json
{
  "host": "<host>",
  "device": "<device>"
}
```
Replace <host> with the IP address or domain name of the Execution Server and <device> with the local device to use for PyTorch operations.

To use the configuration file, run the script with the -c or --config option:
```bash
python node.py -c config.json
```
