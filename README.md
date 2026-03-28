# HumanCrafter

<p align="center">
  <strong>Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.00468"><img src="https://img.shields.io/badge/arXiv-2511.00468-b31b1b.svg" alt="arXiv"></a>
  <a href="https://paulpanwang.github.io/HumanCrafter/"><img src="https://img.shields.io/badge/Project-Page-2ea44f.svg" alt="Project Page"></a>
  <a href="https://github.com/paulpanwang/humancrafter"><img src="https://img.shields.io/badge/Code-GitHub-181717.svg" alt="GitHub"></a>
</p>

HumanCrafter targets the joint modeling of geometry, appearance, and part semantics in a feed-forward framework, combining human geometric priors, self-supervised semantic priors, and an interactive annotation pipeline for building 3D human-part labels.

## News

- `2025-11-01`: HumanCrafter was released on arXiv: [HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation](https://arxiv.org/abs/2511.00468).
- `2025-11-01`: HumanCrafter was accepted to NeurIPS 2025.
- `2025-11-01`: The project page is available at [paulpanwang.github.io/HumanCrafter](https://paulpanwang.github.io/HumanCrafter/).

## Overview

HumanCrafter aims to unify two tasks that are usually handled separately:

- generalizable 3D human reconstruction from a single image
- semantic 3D segmentation of human parts

The released paper describes a framework that:

- introduces geometric priors during reconstruction
- injects self-supervised semantic priors for part-aware reasoning
- leverages pixel-aligned cross-task aggregation
- couples appearance modeling and semantic consistency in a joint objective

<p align="center">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/human3r_teaser.png" width="88%" alt="HumanCrafter teaser">
</p>

## Method

<p align="center">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/network.png" width="95%" alt="HumanCrafter method overview">
</p>

HumanCrafter uses a feed-forward architecture to regress pixel-aligned point maps and semantic 3D Gaussians from a single image, jointly modeling geometry, appearance, and human-part semantics.

## Showcase Videos

We showcase qualitative results from the [project page](https://paulpanwang.github.io/HumanCrafter/), covering semantic body-part segmentation, textured novel-view rendering, and geometric reconstruction. The public page currently presents these demos as visual galleries and downloadable 3D assets.

<p align="center">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/novel/novel_0.png" width="23%" alt="HumanCrafter textural result 1">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/novel/novel_10.png" width="23%" alt="HumanCrafter textural result 2">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/novel/novel_14.png" width="23%" alt="HumanCrafter textural result 3">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/novel/novel_18.png" width="23%" alt="HumanCrafter textural result 4">
</p>

<p align="center">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/3DGS/novel_10.png" width="23%" alt="HumanCrafter geometric result 1">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/3DGS/novel_14.png" width="23%" alt="HumanCrafter geometric result 2">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/3DGS/novel_29.png" width="23%" alt="HumanCrafter geometric result 3">
  <img src="https://paulpanwang.github.io/HumanCrafter/static/images/3DGS/novel_42.png" width="23%" alt="HumanCrafter geometric result 4">
</p>

Selected downloadable meshes from the project page:

- [Ours: sample A](https://paulpanwang.github.io/HumanCrafter/static/images/3DGS/novel_10.ply)
- [Baseline Human3Diffusion: sample A](https://paulpanwang.github.io/HumanCrafter/static/images/3DGS/output_video1.ply)
- [Ours: sample B](https://paulpanwang.github.io/HumanCrafter/static/images/3DGS/novel_29.ply)
- [Baseline Human3Diffusion: sample B](https://paulpanwang.github.io/HumanCrafter/static/images/3DGS/output_video5.ply)

## Project Page

For more qualitative results, mesh downloads, and paper resources, please visit:

- [HumanCrafter Project Page](https://paulpanwang.github.io/HumanCrafter/)
- [HumanCrafter on arXiv](https://arxiv.org/abs/2511.00468)

## Release Status

- [x] Paper
- [x] Project page
- [x] Qualitative demos and downloadable sample assets
- [ ] Cleaned training and inference code release
- [ ] Checkpoints and evaluation scripts
- [ ] Dependency and environment setup instructions

## Citation

If you find HumanCrafter useful in your research, please cite:

```bibtex
@article{pan2025humancrafter,
  title={HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation},
  author={Pan, Panwang and Shen, Tingting and Li, Chenxin and Lin, Yunlong and Wen, Kairun and Zhao, Jingjing and Yuan, Yixuan},
  journal={arXiv preprint arXiv:2511.00468},
  year={2025}
}
```

## License

No license file is included in the current repository snapshot. Until an official license is released, please check with the authors before redistributing code, models, or derived assets.
