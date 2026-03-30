# BridgeDefender

## Requirements

In order to run the code in this repository, you will need to have the XChainDataGen project installed and properly set up. Please follow the instructions in the XChainDataGen repository to install it, then run the `extract`, `decode` and `decode_graph_data` jobs in order to obtain the necessary data for this project.

## Installation

1. Clone the repository:

   ```bash
   git clone
   ```

2. Inside the repository's root directory, create a virtual environment and activate it.
   Make sure you have Python 3.14 installed, as it is the recommended version for this project.
   Alternatively, use a tool like `pyenv` to manage your Python versions and create a virtual
   environment with Python 3.14.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install all the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Encoding Graph Data into PyG Format

To encode graph data into PyG format, run the following command:

```bash
python main.py graph-dataset --out <output_directory> [--bridges <Bridge1> <Bridge2> ...]
```
