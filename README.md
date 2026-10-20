# Retail GCD

Research code for **Reliability-Aware Attribute-Guided Generalized Category
Discovery for Fine-Grained Retail SKU Recognition**. The implementation uses
per-attribute reliability and conflict-aware pair constraints to prevent
visually similar sibling SKUs from collapsing into one novel cluster.

The research design is in [RESEARCH_PROPOSAL.md](RESEARCH_PROPOSAL.md), GPU
instructions are in [TRAINING_HANDOFF.md](TRAINING_HANDOFF.md), and paper/code
attribution is recorded permanently in [PROVENANCE.md](PROVENANCE.md).
The file-by-file review of all three upstream implementations is in
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). The final Vietnamese integration map,
verification status, and criteria for claiming an improvement are in
[INTEGRATION_REPORT.md](INTEGRATION_REPORT.md). The exact dataset decision and
GPU handoff procedure are in
[DATASET_AND_TRAINING_GUIDE.md](DATASET_AND_TRAINING_GUIDE.md).

## Repository layout

```text
configs/                         Main planogram and generated ablation configs
data/                            Manifest example (private/raw data is gitignored)
docs/METHOD.md                   Mathematical method and complete objective
docs/EXPERIMENTS.md              Main comparison and required ablations
scripts/prepare_planogram...     Planogram crop annotation converter
scripts/generate_experiments.py  Multi-seed experiment matrix generator
src/retail_gcd/
  cli/                           Evaluate, extract features, estimate class count
  evaluation/                    K estimation and detailed result reporting
  utils/                         Atomic checkpoints and deterministic seeds
  config.py                      Strict experiment config validation
  data.py                        Manifest-backed known/novel dataset
  model.py                       CLIP encoders, reliability and fusion
  losses.py                      GCD, contrastive, calibration and conflict losses
  metrics.py                     Globally aligned clustering metrics
  train.py                       Reproducible training entry point
tests/                           Core correctness tests
```

## Status

- SelEx, GET, and DebGCD were audited; source URLs, exact commits, conceptual
  influences, and non-equivalences are recorded in `PROVENANCE.md`. Upstream
  trees are not vendored into the final repository.
- Retail CSV manifest validation is implemented.
- CLIP visual/attribute encoders and attribute-level reliability are implemented.
- Reliability calibration, consistency, must-link, and conflict losses are implemented.
- Oracle class-count training and GCD evaluation are implemented.
- Forward/backward and a one-train-batch/one-eval-batch CLI verification passed;
  generated environments/checkpoints are not included in the clean repository.
- Long training has intentionally not been run because this host has no NVIDIA
  GPU. Use the same project on a CUDA host for scientific experiments.

## Environment

Use Python 3.10 or 3.11. Install a CUDA-compatible PyTorch build following the
official PyTorch selector first, then install the project:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install torch torchvision --index-url <PYTORCH_CUDA_INDEX>
pip install -e .
```

Do not blindly use the old environments inside the paper repositories; they
target different CUDA/Python versions.

## Dataset manifest

Copy `data/manifest.example.csv` to `data/manifest.csv`. Each row is an image.
Required fields are:

- `image_path`, relative to the configured image root;
- `sku_id`, used only for labeled supervision or evaluation;
- `split`: `labeled`, `unlabeled`, `val`, or `test`;
- `is_known` and `is_labeled`, each `0` or `1`;
- one value and one confidence column for every attribute, such as `volume` and
  `volume_conf`.

Validate before training:

```bash
retail-gcd-validate data/manifest.csv --image-root data/images
```

Novel targets remain in the manifest for evaluation but are masked from every
supervised loss by `is_labeled`.

For planogram product crops, convert an annotation CSV containing at least
`image_path`, `sku_id`, and `source_split` (`train`, `val`, or `test`):

```bash
.venv/Scripts/python scripts/prepare_planogram_manifest.py \
  --annotations path/to/crop_annotations.csv \
  --output data/processed/planogram_manifest.csv
```

Optional columns are `is_known`, `session_id`, `source`, and each configured
attribute/confidence pair. Missing confidence is treated as zero, never as
oracle certainty.

## Train

Edit `configs/planogram_proposed.json`, especially the paths, CLIP checkpoint, batch
size, and output directory. Then run:

```bash
retail-gcd --config configs/planogram_proposed.json
```

The first run downloads the configured OpenCLIP weights. Every epoch writes a
checkpoint and JSON metrics under `output_dir`.

Set `model.fusion_mode` to `visual`, `attributes`, `simple`, `oracle`, or
`proposed` to run the five pilot/ablation variants without changing source.
`oracle` is meaningful only when its manifest contains manually verified clean
attributes; it must not be reported with ordinary OCR fields renamed as oracle.

Generate the 6-method × 3-seed experiment matrix:

```bash
python scripts/generate_experiments.py
```

Offline checkpoint evaluation and estimated-K protocol:

```bash
retail-gcd-eval --checkpoint outputs/planogram/proposed/seed_1/best.pt
retail-gcd-extract --checkpoint outputs/planogram/proposed/seed_1/best.pt \
  --splits labeled unlabeled --output outputs/features_seed1.npz
retail-gcd-estimate-k --features outputs/features_seed1.npz \
  --min-classes 100 --max-classes 300 --step 5 \
  --output outputs/class_count_seed1.json
```

## Experimental order

1. Visual-only.
2. OCR/attributes-only.
3. Simple visual-text fusion.
4. Oracle clean attributes.
5. Proposed per-attribute reliability and conflict-aware model.

Do not start the full run until manifest leakage and duplicate images have been
audited. Report natural-OCR and leakage-controlled protocols separately.

