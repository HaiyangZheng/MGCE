# Generalized Fine-Grained Category Discovery with Multi-Granularity Conceptual Experts

<p align="center">
    <a href="https://arxiv.org/abs/2509.26227"><img src="https://img.shields.io/badge/arXiv-2509.26227-red"></a>
</p>
<p align="center">
	Generalized Fine-Grained Category Discovery with Multi-Granularity Conceptual Experts<br>
</p>

![framework](assets/framework.png)
Generalized Category Discovery (GCD) is an open-world problem that clusters unlabeled data by leveraging knowledge from partially labeled categories. A key challenge is that unlabeled data may contain both known and novel categories. Existing approaches suffer from two main limitations. First, they fail to exploit multi-granularity conceptual information in visual data, which limits representation quality. Second, most assume that the number of unlabeled categories is known during training, which is impractical in real-world scenarios. To address these issues, we propose a Multi-Granularity Conceptual Experts (MGCE) framework that adaptively mines visual concepts and integrates multi-granularity knowledge for accurate category discovery. MGCE consists of two modules: (1) Dynamic Conceptual Contrastive Learning (DCCL), which alternates between concept mining and dual-level representation learning to jointly optimize feature learning and category discovery; and (2) Multi-Granularity Experts Collaborative Learning (MECL), which extends the single-expert paradigm by introducing additional experts at different granularities and by employing a concept alignment matrix for effective cross-expert collaboration. Importantly, MGCE can automatically estimate the number of categories in unlabeled data, making it suitable for practical open-world settings. Extensive experiments on nine fine-grained visual recognition benchmarks demonstrate that MGCE achieves state-of-the-art results, particularly in novel-class accuracy. Notably, even without prior knowledge of category numbers, MGCE outperforms parametric approaches that require knowing the exact number of categories, with an average improvement of 3.6\%.

## 🚀 Quick Start

### Environment Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate gcd-env
```

2. Configure data paths:
```bash
# Modify the data directory paths in config.py
vim config.py
```

### Training Pipeline

For each dataset, follow these two steps:

1. **Get KNN param:**
```bash
bash scripts_knn/[dataset_name].sh
```

2. **Train the model:**
```bash
bash scripts/train_[dataset_name].sh
```

Replace `[dataset_name]` with one of: `cub`, `scars`, `animalia`, `fungi`, `mollusca`, `actinopterygii`, `herbarium19`, `nabirds`, `reptilia`.

## 📊 Results

### Known Number of Categories

| Dataset | Seed | All | Old | New |
|---------|------|-----|-----|-----|
| **CUB** | 0 | 70.6 | 76.2 | 67.8 |
| | 1 | 70.3 | 72.6 | 69.1 |
| | 2 | 70.2 | 73.4 | 68.6 |
| | **Avg** | **70.37** | **74.07** | **68.50** |
| **SCARS** | 0 | 61.7 | 75.8 | 54.9 |
| | 1 | 61.4 | 76.0 | 54.3 |
| | 2 | 61.3 | 75.8 | 54.4 |
| | **Avg** | **61.47** | **75.87** | **54.53** |
| **Animalia** | 0 | 59.2 | 60.6 | 58.6 |
| | 1 | 58.7 | 62.1 | 57.3 |
| | 2 | 60.8 | 61.9 | 60.3 |
| | **Avg** | **59.57** | **61.53** | **58.73** |
| **Fungi** | 0 | 58.1 | 68.2 | 53.6 |
| | 1 | 58.4 | 68.0 | 54.2 |
| | 2 | 58.2 | 69.7 | 53.1 |
| | **Avg** | **58.23** | **68.63** | **53.63** |
| **Mollusca** | 0 | 54.9 | 60.9 | 51.7 |
| | 1 | 55.3 | 58.5 | 53.6 |
| | 2 | 56.3 | 65.7 | 51.3 |
| | **Avg** | **55.50** | **61.70** | **52.20** |
| **Actinopterygii** | 0 | 45.3 | 57.5 | 40.6 |
| | 1 | 45.0 | 52.1 | 42.2 |
| | 2 | 45.5 | 53.8 | 42.3 |
| | **Avg** | **45.27** | **54.47** | **41.70** |
| **Herbarium 19** | 0 | 45.3 | 53.1 | 41.1 |
| | 1 | 45.3 | 53.7 | 40.8 |
| | 2 | 45.4 | 52.7 | 41.4 |
| | **Avg** | **45.33** | **53.17** | **41.10** |
| **NABirds** | 0 | 50.8 | 73.9 | 40.6 |
| | 1 | 50.6 | 73.5 | 40.3 |
| | 2 | 47.7 | 72.1 | 36.9 |
| | **Avg** | **49.70** | **73.17** | **39.27** |
| **Reptilia** | 0 | 30.5 | 45.4 | 24.4 |
| | 1 | 30.2 | 46.4 | 23.6 |
| | 2 | 30.9 | 45.4 | 25.1 |
| | **Avg** | **30.53** | **45.73** | **24.37** |

### Unknown Number of Categories

| Dataset | Estimated K | Seed | All | Old | New |
|---------|-------------|------|-----|-----|-----|
| **CUB** | 202 | 0 | 68.9 | 72.6 | 67.1 |
| | | 1 | 68.1 | 73.8 | 65.2 |
| | | 2 | 67.2 | 73.2 | 64.3 |
| | | **Avg** | **68.07** | **73.20** | **65.53** |
| **SCARS** | 203 | 0 | 58.6 | 74.6 | 51.0 |
| | | 1 | 57.2 | 73.5 | 49.4 |
| | | 2 | 57.6 | 70.1 | 51.6 |
| | | **Avg** | **57.80** | **72.73** | **50.67** |
| **Animalia** | 53 | 0 | 61.0 | 65.5 | 59.1 |
| | | 1 | 60.1 | 66.0 | 57.7 |
| | | 2 | 59.8 | 61.2 | 59.2 |
| | | **Avg** | **60.30** | **64.23** | **58.67** |
| **Fungi** | 112 | 0 | 55.8 | 71.3 | 48.9 |
| | | 1 | 56.7 | 71.6 | 50.0 |
| | | 2 | 54.9 | 74.3 | 46.4 |
| | | **Avg** | **55.80** | **72.40** | **48.43** |
| **Mollusca** | 43 | 0 | 50.9 | 59.7 | 46.2 |
| | | 1 | 51.4 | 62.8 | 45.3 |
| | | 2 | 50.9 | 61.7 | 45.1 |
| | | **Avg** | **51.07** | **61.40** | **45.53** |
| **Actinopterygii** | 61 | 0 | 43.4 | 55.2 | 38.8 |
| | | 1 | 45.3 | 54.7 | 41.5 |
| | | 2 | 43.2 | 55.4 | 38.4 |
| | | **Avg** | **43.97** | **55.10** | **39.57** |
| **Herbarium 19** | 459 | 0 | 43.5 | 53.1 | 38.4 |
| | | 1 | 43.1 | 53.1 | 37.7 |
| | | 2 | 42.6 | 54.6 | 36.5 |
| | | **Avg** | **43.07** | **53.60** | **37.37** |
| **NABirds** | 715 | 0 | 53.8 | 74.3 | 44.7 |
| | | 1 | 54.3 | 75.0 | 45.1 |
| | | 2 | 51.8 | 74.9 | 41.6 |
| | | **Avg** | **53.30** | **74.73** | **43.80** |
| **Reptilia** | 140 | 0 | 29.5 | 52.8 | 20.1 |
| | | 1 | 30.1 | 46.1 | 23.7 |
| | | 2 | 29.8 | 52.8 | 20.5 |
| | | **Avg** | **29.80** | **50.57** | **21.43** |

*Note: "All" refers to overall accuracy, "Old" to known categories, and "New" to novel categories.*


## Acknowledgments

This repository builds upon the excellent work of:
- [DCCL](https://github.com/TPCD/DCCL)
- [SimGCD](https://github.com/CVMI-Lab/SimGCD)

We thank the authors for their contributions to the field of Generalized Category Discovery.
