# Model-Based Uncertainty in Value Functions

Official PyTorch implementation of the AISTATS 2023 paper ["Model-Based Uncertainty in Value
Functions"](https://arxiv.org/abs/2302.12526).

## Installation

Prerequisites:
- `conda` (optional, install for Option #1 below)
- `docker` (optional, install for  Option #2 below)
- `swig` - `sudo apt-get -y install swig`

### Option #1: `conda` environment -- no Mujoco support.

1. Clone the repository and `cd` into it
```bash
git clone https://github.com/boschresearch/ube-mbrl.git && cd ube-mbrl
```
2. Create a conda environment
```bash
conda env create --file=environment.yml
```
3. Activate the environment and install the package + dependencies
 ```bash
conda activate ube_mbrl
pip install -e .
```
4. If you want to test on Mujoco environments, you may install it following commands in the
   [Dockerfile](docker/Dockerfile).

### Option #2: Docker container.
Make sure `docker` is installed and configured.

1. Build docker image. The Dockerfile installs Mujoco and Pytorch with CUDA support (you may need to
   change the CUDA version depending on your GPU).
```bash
cd docker/
./build_docker.sh
```
2. After the image is created, you can access it via
```bash
docker run --rm -ti ube-mbrl
```

## Usage

### Running tabular RL experiments
Relevant config file for these experiments is [here](ube_mbrl/tabular/config.py).
```bash
cd {path_to_repo}/ube_mbrl

# Toy example
python tabular/toy_mdp_example.py

# DeepSea environment
python tabular/deep_sea_exp.py

# 7-room environment
python tabular/nroom_exp.py
```

### Running continuous control experiments
The model learning config is specified in this [YAML file](ube_mbrl/conf/mbrl_lib_config.yaml).
```bash
cd {path_to_repo}/ube_mbrl

# SAC agent - config in `ube_mbrl/conf/config_sac_online.py`
python train_scripts/train_sac_online.py

# Model-based SAC agent with UBE exploration signal - config in `ube_mbrl/conf/config_qusac_online.py`
python train_scripts/train_qusac_online.py
```
### Reproducing paper plots
We provide the experiment [data](data) that can be used with the provided [Jupyter
notebooks](notebooks) to reproduce the figures in the paper.

## Citation
```
@InProceedings{luis_model-based_2023,
  title = 	 {Model-{Based} {Uncertainty} in {Value} {Functions}},
  author =       {Luis, Carlos E. and Bottero, Alessandro G. and Vinogradska, Julia and Berkenkamp, Felix and Peters, Jan},
  booktitle = 	 {Proceedings of the Twenty Sixth International Conference on Artificial Intelligence and Statistics},
  year = 	 {2023},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}
```


## License
The code is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.
