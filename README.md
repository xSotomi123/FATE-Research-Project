# FATE Experiment Source Code

This is part of the 2025/Q4 [Research Project](https://github.com/TU-Delft-CSE/Research-Project) of [TU Delft](https://github.com/TU-Delft-CSE) and contains the source code used to assess the integration complexity and usability of [FATE](https://github.com/FederatedAI/FATE).

## Installation guide:

This framework was installed by following FATE's [quick start guide](https://fate.readthedocs.io/en/latest/2.0/fate/quick_start/). First create a clean virtual environment:

```
conda create -n fate_env python=3.10 -y
```
```
conda activate fate_env
```

When executing the code, we ran into some issues that required specific versions of PyTorch, NumPy, and Transformers:

```
pip install torch==2.1.0
```
```
pip install numpy==1.26.4
```
```
pip install transformers==4.36.2
```

Now install FATE Client:

```
python -m pip install -U pip && python -m pip install fate_client[fate,fate_flow]==2.1.1
```

After successfully installing the FATE Client, initialize the FATE-Flow service and FATE Client:

```
mkdir fate_workspace
```
```
fate_flow init --ip 127.0.0.1 --port 9380 --home $(pwd)/fate_workspace
```
```
pipeline init --ip 127.0.0.1 --port 9380
```
```
fate_flow start
```
```
fate_flow status
```

Download the example data that was used:

```
wget https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_guest.csv && wget https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_host.csv
```

Now just run the `hetero_fl_lr_tutorial_test.py` file to create the same pipeline used in our experiment.
