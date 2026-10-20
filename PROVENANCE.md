# Code and idea provenance

This file records how the repository relates to the research code downloaded
during development. It must remain in the repository even after upstream source
trees are removed.

## Summary

The production code under `src/retail_gcd/` is a clean implementation written
for this repository. No Python file from SelEx, GET, or DebGCD was copied into
the package. The official repositories were inspected to understand their
interfaces, losses, evaluation conventions, and reproducibility requirements.

Consequently:

- this repository does **not** claim SelEx, GET, or DebGCD as original work;
- results obtained by running their official implementations must cite them;
- results from this implementation must describe the concepts reused below;
- the proposed attribute-level reliability and conflict-aware affinity modules
  are local code, but novelty still has to be established by literature review
  and experiments.

## Upstream sources inspected

| Work | Repository | Inspected commit | License |
|---|---|---|---|
| SelEx, ECCV 2024 | https://github.com/SarahRastegar/SelEx | `569ee7085e779999502bd73ea92240f3d32fc84d` | MIT |
| GET, CVPR 2025 | https://github.com/enguangW/GET | `8e908c061660ce95b07596b9a7f190e5c444f7cb` | MIT |
| DebGCD, ICLR 2025 | https://github.com/Visual-AI/DebGCD | `9ec30fe3d54dfccd9934a862cf5cb9ff05de6f05` | MIT |

## Component-level mapping

| Local component | File | Origin/influence | What is local |
|---|---|---|---|
| Two-view GCD training | `src/retail_gcd/train.py` | SimGCD-style training pattern, also used by GET and related GCD work | Configuration, manifest contract, scheduling, resume, checkpoint metadata |
| Instance contrastive loss | `src/retail_gcd/losses.py` | Standard SimCLR/InfoNCE; used broadly in GCD including SimGCD/SelEx-family implementations | Clean local implementation |
| Supervised contrastive loss | `src/retail_gcd/losses.py` | SupCon and standard GCD representation learning | Clean local implementation with labeled-row masking |
| Cross-view soft pseudo-label consistency | `src/retail_gcd/losses.py` | DINO/SimGCD/GET family of self-distillation objectives | Confidence warm-up and local implementation |
| Mean-entropy regularization | `src/retail_gcd/losses.py` | Common GCD anti-collapse objective; present in GET/SimGCD-style code | Clean local implementation |
| Multimodal visual/text representation | `src/retail_gcd/model.py` | GET motivates visual-semantic dual information through CLIP | This repo uses observed OCR/structured attributes, not GET's TES pseudo-token generator |
| Attribute-level reliability | `src/retail_gcd/model.py` | Research proposal; DebGCD motivates treating confidence/bias explicitly | Local per-sample, per-attribute estimator combining OCR confidence, presence, and cross-modal agreement |
| Conflict-aware pair loss | `src/retail_gcd/losses.py` | Proposed retail-specific mechanism | Local must-link/cannot-link construction and margin loss |
| Hierarchical self-expertise | `evaluation/hierarchy.py`, `losses.py`, `train.py` | SelEx semi-supervised hierarchical clustering and pseudo-label expertise | Clean fixed-label K-means adaptation and level-decayed hierarchy loss |
| Dual visual/attribute learning and CICO | `model.py`, `losses.py` | GET dual visual/pseudo-text branches and anchor consistency | TES is replaced by observed retail OCR/attributes; local CICO implementation |
| Semantic distribution learning | `model.py`, `losses.py` | DebGCD OVA distribution detector | Local OVA head, supervised/entropy/consistency losses |
| Auxiliary debiased learning | `model.py`, `losses.py` | DebGCD auxiliary classifier and certainty-weighted pseudo labels | Local debiased head integrated with retail fusion |
| Global Hungarian evaluation | `src/retail_gcd/metrics.py` | Standard clustering evaluation used throughout GCD | Local implementation; one assignment is shared by All/Old/New to prevent inflated subset scores |
| Distribution/imbalance awareness | experiment plan | DebGCD | No DebGCD distribution-guidance module is currently copied or claimed as implemented |

## Important non-equivalences

The local method is not a drop-in reproduction of any paper:

- It implements a SelEx-adapted hierarchy, not upstream confusion smoothing exactly.
- It does not implement GET's Text Embedding Synthesizer (TES).
- It implements the central DebGCD SDL/ADL mechanisms with a different backbone and retail inputs.
- `fusion_mode=visual` is a controlled local baseline, not an official paper reproduction.

Official paper numbers must therefore be reproduced with the official repositories
or clearly labeled as numbers reported by the papers. Never label a local ablation
as “SelEx”, “GET”, or “DebGCD”.

## Dataset provenance

No third-party dataset is distributed with this repository. The primary paper
protocol requires RetailVision/planogram product crops. Every external public
dataset used in experiments must be cited and recorded with its exact release,
license, preprocessing, and split.

