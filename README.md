# AAA (Automated Accessibility Assessment)

<div align="center">

</div>

## Dataset

**⚠️ Important: Please download at least the TPS dataset and place it in the `data/TPS` directory at the same level as this project before running the scripts.**

The dataset consists of four parts: **APR**, **CCT**, **TPS**, and **CPE**. 

**Note**: In our experiments, we only use the **TPS** dataset. 

All datasets are available at: <https://zenodo.org/records/17548393>

## Installation

### Requirements
- Python >= 3.8
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Run installation scripts
bash .ci/install-dev.sh
bash .ci/install.sh

# Activate virtual environment
source .env/bin/activate
```

## Usage

### Clustering Scripts
Execute different clustering approaches:

- **GRASP with Custom GNN**: `bash scripts/grasp.sh`
- **GRASP with IGNN**: `bash scripts/ignn.sh`
- **SDC (Structure-Dependent Clustering)**: `bash scripts/sdc.sh`

### Similarity Calculation Scripts
Compute similarity metrics for evaluation:

- **GRASP Cosine Similarity**: `bash scripts/cos.sh`
- **IGNN Cosine Similarity**: `bash scripts/ignn_cos.sh`
- **SDC Cosine Similarity**: `bash scripts/sdc_cos.sh`

## Requirements

See [requirements-dev.txt](./requirements-dev.txt), [requirements.txt](./requirements.txt) and [pyproject.toml:dependencies](./pyproject.toml).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).
