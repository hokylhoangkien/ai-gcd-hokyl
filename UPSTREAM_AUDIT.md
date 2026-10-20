# Upstream source audit

This audit was produced from the exact commits below, not from README files
alone. The source trees were downloaded into the ignored `.upstream_audit/`
directory while the integration was implemented and tested.

| Repository | Commit | Python files | Approx. Python lines |
|---|---|---:|---:|
| SelEx | `569ee7085e779999502bd73ea92240f3d32fc84d` | 30 | 5,321 |
| GET | `8e908c061660ce95b07596b9a7f190e5c444f7cb` | 20 | 3,380 |
| DebGCD | `9ec30fe3d54dfccd9934a862cf5cb9ff05de6f05` | 18 | 3,320 |

The counts include benchmark dataset loaders and bundled backbone code. They
are useful context but are not a quality metric by themselves.

## SelEx audit

Primary files read:

- `methods/contrastive_training/contrastive_training.py`
- `methods/clustering/faster_mix_k_means_pytorch.py`
- `methods/clustering/extract_features.py`
- `methods/clustering/k_means.py`
- `methods/estimate_k/estimate_k.py`
- `project_utils/cluster_and_log_utils.py`
- `project_utils/cluster_utils.py`
- `data/get_datasets.py`

Mechanism traced through the training path:

1. DINO ViT feature extractor with only late blocks trainable.
2. Balanced sampling between labeled and unlabeled examples.
3. Periodic mixed/semi-supervised K-means over the merged training set.
4. Progressive coarsening of labeled and novel cluster counts.
5. Prototype and cluster-radius computation per hierarchy level.
6. Confusion-aware smoothing of InfoNCE negatives.
7. Supervised contrastive loss at the base level and pseudo-label supervised
   contrastive losses on lower-dimensional slices at coarser levels.
8. K-means evaluation and optional class-number estimation.

Local adaptation:

- `evaluation/hierarchy.py` implements fixed-label semi-supervised K-means,
  prototype/radius extraction, and progressive known/novel hierarchy coarsening.
- `losses.py::hierarchical_contrastive_loss` applies level-decayed self-expertise.
- `train.py::refresh_hierarchy` refreshes global assignments periodically.

Not ported exactly:

- SelEx confusion-factor label smoothing based on pair distance versus cluster
  radii is not yet identical to upstream.
- Feature-dimension slicing at each hierarchy level is replaced by full projected
  retail features.
- OpenCLIP replaces the upstream DINO implementation in the main configuration.

Therefore the local baseline is named **SelEx-adapted**, never official SelEx.

## GET audit

Primary files read:

- `s1_TES.py`
- `s2_GET.py`
- `model.py`
- `clip/model.py`
- `clip/clip.py`
- `util/general_utils.py`
- `util/cluster_and_log_utils.py`
- `data/get_datasets.py`

Mechanism traced through the two stages:

1. Stage 1 projects CLIP image features into several pseudo-word tokens.
2. A modified frozen CLIP text transformer encodes those pseudo words.
3. TES is trained with bidirectional image/text alignment and base-class text
   distillation from actual class-name prompts.
4. Stage 2 freezes TES, trains separate visual and pseudo-text branches with
   supervised CE, cross-view distillation, mean entropy, InfoNCE, and SupCon.
5. CICO constructs labeled-class anchors in both modalities and applies
   bidirectional cross-modal distribution consistency.
6. Evaluation reports image, text, and probability-averaged predictions.

Local adaptation:

- `model.py` exposes visual, observed-attribute, fused, debiased, and distribution
  outputs with a shared cluster head for comparable assignments.
- `losses.py::cross_modal_anchor_consistency` ports the CICO anchor-distribution
  idea to observed OCR/structured attributes.
- Both visual and attribute branches receive supervised and cross-view losses.

Why TES is not copied:

Retail product crops can provide observed OCR fields such as brand, variant,
and volume. Generating pseudo text from the image would discard the central
signal and duplicate visual evidence. The replacement is scientifically
intentional, but means this is **GET-adapted**, not official GET.

## DebGCD audit

Primary files read:

- `train_DebGCD.py`
- `model.py`
- `util/cluster_and_log_utils.py`
- `util/osr_utils.py`
- `util/general_utils.py`
- `data/get_datasets.py`

Mechanism traced through the training path:

1. SimGCD head and representation objectives form the base loss.
2. An OVA semantic distribution head predicts known/unknown status relative to
   each known category.
3. Labeled OVA loss combines positive-category likelihood and hardest negative.
4. Unlabeled SDL uses OVA entropy minimization and cross-view consistency.
5. Distribution certainty is derived from OVA unknown probability.
6. A separate debiased classifier receives labeled CE plus high-confidence
   pseudo-label CE weighted by distribution certainty.
7. Evaluation compares the primary and debiased classifiers and includes a
   bias-matrix diagnostic.

Local adaptation:

- `model.py` contains separate primary, OVA distribution, and debiased heads.
- `losses.py` implements OVA supervised loss, OVA entropy, SDL consistency,
  distribution certainty, and certainty-weighted auxiliary debiased learning.
- Known SKU indices are forced to occupy `[0, K_known)` in the dataset mapping,
  which is required by the OVA target layout.

The local implementation uses a CLIP retail backbone and different checkpoint
format, so the baseline is **DebGCD-adapted**, not official DebGCD.

## Integration status

The local `proposed` configuration combines:

- SimGCD-style supervised/unsupervised representation and cluster learning;
- SelEx-adapted hierarchical self-expertise;
- GET-adapted dual visual/attribute learning and CICO;
- DebGCD-adapted SDL and auxiliary debiased learning;
- the proposed per-attribute reliability and conflict-aware pair constraints.

Each family also has an isolated generated configuration so improvements can be
attributed through ablations. Passing a smoke test proves code-path integrity,
not research superiority; only multi-seed planogram experiments can establish
that.

