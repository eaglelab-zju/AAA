# AAA

Official implementation and datasets of the **AAAI 2026 Special Track AI for Social Impact** paper:  

[**Towards Scalable Web Accessibility Audit with MLLMs as Copilots**](https://arxiv.org/abs/2511.03471).

This repository provides the implementation of the **GRASP** method and accompanying datasets **AWA**.

## 📊 Dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17548393.svg)](https://doi.org/10.5281/zenodo.17548393)

The **AWA Web Accessibility Benchmark** consists of four datasets:

- **APR** – Accessibility-relevant Page Recognition
- **CCT** – CAPTCHA of Cognitive Tests 
- **TPS** – Triple-representativeness Page Sampling
- **CPE** – Complete Process Extraction 

All datasets are publicly available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17548393.svg)](https://doi.org/10.5281/zenodo.17548393)

> [!Important]
> Before running any scripts, please download at least the **TPS** dataset and place it in the `data/TPS` directory at the same level as this project.  
> 
> This repository implements the **GRASP** method, which uses only the TPS dataset.  
> 
> For experiments involving **APR**, **CCT**, and **CPE**, please refer to the prompts and run them with any MLLMs as in the [Appendix of our paper](https://arxiv.org/abs/2511.03471).


## 🛠️ Installation

**Requirements:**
- Python >= 3.8
- CUDA-compatible GPU (recommended)
- Dependencies are listed in:
  * [requirements.txt](./requirements.txt)
  * [requirements-dev.txt](./requirements-dev.txt)
  * [pyproject.toml](./pyproject.toml)

**Setup:**
```bash
# Run installation scripts
bash .ci/install-dev.sh
bash .ci/install.sh

# Activate virtual environment
source .env/bin/activate
```

## 🚀 Usage

Please run the experiments of GRASP and SDC as follows.

**Page Sampling:**

Run different clustering approaches for the TPS dataset:

* **GRASP with GCN**:

  ```bash
  bash scripts/grasp.sh
  ```
* **GRASP with IGNN**:

  ```bash
  bash scripts/ignn.sh
  ```
* **SDC w & w/o TSNE**:

  ```bash
  bash scripts/sdc.sh
  ```

**Evaluation:**

Compute metrics for evaluation:

* **GRASP with GCN**:

  ```bash
  bash scripts/cos.sh
  ```
* **GRASP with IGNN**:

  ```bash
  bash scripts/ignn_cos.sh
  ```
* **SDC w & w/o TSNE**:

  ```bash
  bash scripts/sdc_cos.sh
  ```

## 🤝 Contributing

Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on reporting issues, contributing improvements, and extending the GRASP framework.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{aaa,
title={Towards Scalable Web Accessibility Audit with {MLLMs} as Copilots},
author={Ming Gu and Ziwei Wang and Sicen Lai and Zirui Gao and Sheng Zhou and Jiajun Bu},
journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
year={2026},
url={https://arxiv.org/abs/2511.03471}, 
}
```
