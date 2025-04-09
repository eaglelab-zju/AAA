# AAA

<div align="center">

<!-- [![Latest Release](https://img.shields.io/github/v/tag/galogm/py_setting)](https://github.com/galogm/py_setting/tags) -->

</div>

## Clone

```
git clone --recurse-submodules git@github.com:eaglelab-zju/AAA.git
```

## Installation

- python>=3.8
- for installation scripts see `.ci/install-dev.sh`, `.ci/install.sh`

```bash
bash .ci/install-dev.sh
bash .ci/install.sh
```

## Usage

Scripts for clustering data sets in different ways:

- grasp-custom-GNN: `scripts/grasp.sh`

- grasp-IGNN: `scripts/ignn.sh`

- sdc: `scripts/sdc.sh`


Script for calculating similarity:

- grasp-custom-GNN: `scripts/cos.sh`

- grasp-IGNN: `scripts/ignn_cos.sh`

- sdc: `scripts/sdc_cos.sh`

## Requirements

See [requirements-dev.txt](./requirements-dev.txt), [requirements.txt](./requirements.txt) and [pyproject.toml:dependencies](./pyproject.toml).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).
