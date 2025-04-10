# AAA

<div align="center">

<!-- [![Latest Release](https://img.shields.io/github/v/tag/galogm/py_setting)](https://github.com/galogm/py_setting/tags) -->

</div>

## Clone

```
git clone git@github.com:eaglelab-zju/AAA-UIST.git
```

## Dataset
Because the domData dataset is too large, it was downloaded in two batches.

- APR: <https://drive.google.com/drive/folders/1gCa2_wLTK4EZm1KCsjt4oy7jWYRwz2bJ?usp=sharing>

- CCT: <https://drive.google.com/drive/folders/1UhCEH2kAsYA1w9nBjPoOS9ZD91362Tkg?usp=sharing>

- TPS:

  - TPS-graphData: <https://drive.google.com/file/d/1ywnKWHg2cN9NX7SXM8Rd48EY-RMq6_op/view?usp=sharing>

  - TPS-axeData: <https://drive.google.com/file/d/1IzQF0psfFv0uhLWcY7ayzLGEy4VINQ47/view?usp=sharing>

  - TPS-domData:
    - batch1: <https://drive.google.com/file/d/14igtp7uQ8GFEoWfgtIyP1UKUlV-6Oeaf/view?usp=sharing>
    - batch2: <https://drive.google.com/file/d/1BMRLPkQXN9ziAfCldjrzNvEtxPwCtiFt/view?usp=sharing>


## Installation

- python>=3.8
- for installation scripts see `.ci/install-dev.sh`, `.ci/install.sh`

```bash
bash .ci/install-dev.sh
bash .ci/install.sh
```

## Usage

Scripts for clustering data sets in different ways:

`source .env/bin/activate`

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
