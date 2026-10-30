
# RAGeo: Residual-aware Global Representation Learning for Robust Cross-View Geo-Localization

**Official repository for our paper accepted by ISPRS Journal of Photogrammetry and Remote Sensing (2026).**

**Youyuan Xue, Fan Guo, Jiahua Zou, Jin Tang, Zhihu Wu**

[![Paper](https://img.shields.io/badge/Paper-ISPRS_JPRS-blue)](https://doi.org/10.1016/j.isprsjprs.2026.10.033)
![Code](https://img.shields.io/badge/Code-Coming_Soon-orange)

**Paper:** https://doi.org/10.1016/j.isprsjprs.2026.10.033

---

## Overview

Cross-view geo-localization (CVGL) aims to retrieve the geographically corresponding satellite image for a ground-level query. While global-descriptor retrieval enables efficient large-scale matching, its performance can deteriorate considerably under unknown orientations and restricted fields of view (FoVs).

In this work, we investigate whether intermediate visual features retain useful spatial correspondence evidence that is not fully captured by the final global descriptor.

We propose **RAGeo**, a residual-aware global representation learning framework that integrates reliable spatial evidence into compact global representations during training.

RAGeo combines:

- **Spatial-Strip Contrastive Supervision:** Learning spatial correspondence across different visibility conditions.
- **FoV-Balanced Spatial Supervision:** Strengthening spatial learning under restricted fields of view.
- **Margin-Aware Spatial Distillation:** Transferring useful spatial evidence into the global descriptor while accounting for retrieval margins.

At inference time, RAGeo retains the original global-descriptor retrieval pipeline without requiring feature-map storage, local matching, or top-K reranking.

## Framework

The framework illustration will be added here.

## Datasets

RAGeo is evaluated on three widely used cross-view geo-localization benchmarks:

- **CVUSA**
- **CVACT**
- **VIGOR**

The experiments cover multiple FoV settings, unknown orientations, and both Same-Area and Cross-Area evaluation protocols.

## Code Release

**The implementation is currently being prepared for public release.**

We plan to provide:

- Model implementation and training configurations
- Training and evaluation scripts
- Instructions for reproducing the reported experiments
- Pretrained model checkpoints, where redistribution is permitted

Please refer to this repository for future updates.

## Citation

If you find this work useful, please cite:

**Youyuan Xue, Fan Guo, Jiahua Zou, Jin Tang, and Zhihu Wu.**  
RAGeo: Residual-aware global representation learning for robust cross-view geo-localization.  
*ISPRS Journal of Photogrammetry and Remote Sensing*, 2026.

DOI: https://doi.org/10.1016/j.isprsjprs.2026.10.033

## Acknowledgments

This work was supported by the Scientific Research Project of Xiang Jiang Lab (No. 23XJ02002) and the Natural Science Foundation of Hunan Province of China (No. 2025JJ50396).

Computing resources were provided in part by the High Performance Computing Center of Central South University.
