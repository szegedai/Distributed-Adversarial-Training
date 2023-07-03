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
