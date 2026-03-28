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

## Method

<p align="center">
  <img src="assets/readme/method-overview.png" width="95%" alt="HumanCrafter method overview">
</p>

HumanCrafter uses a feed-forward architecture to regress pixel-aligned point maps and semantic 3D Gaussians from a single image, jointly modeling geometry, appearance, and human-part semantics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
pip install -e .
```

If you need a CUDA-specific PyTorch build, replace the `pip install torch torchvision` line with the wheel command recommended by the official PyTorch installation guide for your CUDA version.

The repository requirements are expanded with the current HumanCrafter scaffold needs plus commonly used research dependencies referenced from the [PartCrafter requirements](https://github.com/wgsxm/PartCrafter/blob/main/settings/requirements.txt), including:

- `diffusers`, `transformers`, `peft`
- `einops`, `jaxtyping`, `typeguard`
- `opencv-python`, `trimesh`, `pyrender`, `scikit-image`
- `gpustat`, `nvitop`, `wandb[media]`

## Inference

Prepare a checkpoint path inside `configs/inference.yaml`, then run:

```bash
python scripts/inference_humancrafter.py \
  --config configs/inference.yaml \
  --input_dir path/to/images \
  --output_dir outputs/inference
```

For backwards compatibility, the old root entry point still works:

```bash
python inference.py --config configs/inference.yaml --input_dir path/to/images --output_dir outputs/inference
```

Each sample output directory stores:

- `results.npz`
- per-key `.npy` arrays
- `metadata.json`
- `semantic_preview.png` when semantic predictions are available

## Evaluation

Create a manifest following [datasets/README.md](datasets/README.md), then run:

```bash
python scripts/evaluate_humancrafter.py \
  --config configs/evaluation.yaml \
  --manifest datasets/example_manifest.json \
  --output_dir outputs/evaluation \
  --save_predictions
```

The evaluation command writes aggregate metrics to `metrics.yaml` and per-sample metrics to `metrics_per_sample.yaml`.

## Notes

- The architecture in this repository is a clean baseline scaffold derived from the paper description and project-page claims.
- Dataset specifics, full training recipes, and released checkpoints still need to be filled in with project-specific assets.
- The repo is now organized so those components can be added without rewriting the surrounding tooling.

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
