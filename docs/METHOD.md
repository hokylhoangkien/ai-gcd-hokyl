# Method specification

## Problem definition

Let the labeled set be

\[
\mathcal{D}_L=\{(x_i,y_i,a_i,c_i)\}_{i=1}^{N_L},\qquad y_i\in\mathcal{Y}_K,
\]

and the unlabeled set be

\[
\mathcal{D}_U=\{(x_i,a_i,c_i)\}_{i=1}^{N_U},
\]

where the latent labels of `D_U` belong to known and novel classes
\(\mathcal{Y}_K\cup\mathcal{Y}_N\). Here \(a_i^m\) is the observed value of
attribute \(m\), and \(c_i^m\in[0,1]\) is upstream evidence such as OCR
confidence. Novel labels are stored in manifests only for evaluation and are
masked from all supervised objectives.

## Encoders and projection

For two augmentations \(x_i^1,x_i^2\), the CLIP visual encoder and trainable
projection produce

\[
v_i^q=\operatorname{norm}(W_v f_v(x_i^q)), \quad q\in\{1,2\}.
\]

Each observed attribute is prompted with its field name (for example,
`volume: 330 ml`) and encoded independently:

\[
t_i^m=\operatorname{norm}(W_t f_t(\operatorname{prompt}(m,a_i^m))).
\]

Missing attributes use a generic missing-value prompt and are assigned a zero
presence mask.

## Attribute-level reliability

The reliability head uses image evidence, attribute evidence, their cosine
agreement, OCR confidence, and presence:

\[
r_i^m=\sigma\left(g_\theta\left[
v_i;t_i^m;\cos(v_i,t_i^m);c_i^m;p_i^m
\right]\right)p_i^m.
\]

Normalized weights and the fused representation are

\[
\alpha_i^m=\frac{r_i^m}{\sum_j r_i^j+\epsilon},\quad
\bar t_i=\sum_m\alpha_i^m t_i^m,
\]

\[
z_i=\operatorname{norm}\left(v_i+\bar r_i\bar t_i\right),\qquad
\bar r_i=\frac{1}{M}\sum_m r_i^m.
\]

The `visual`, `attributes`, `simple`, and `oracle` modes replace this fusion in
controlled ablations while keeping the encoder, split, and optimizer fixed.

## Cluster prediction

With normalized prototype weights \(W\), the cluster logits are

\[
\ell_i=Wz_i/\tau.
\]

In the oracle-K protocol, output dimension is the number of classes present in
the training manifest. Estimated-K is computed separately and must never be
mixed with oracle-K in the same result column.

## SelEx-adapted hierarchical self-expertise

At each configured refresh interval, features for the merged labeled and
unlabeled training set are extracted without augmentation. Level-zero
semi-supervised K-means fixes labeled assignments to their known indices:

\[
q_i=\arg\min_k\lVert z_i-\mu_k\rVert_2^2,\qquad
q_i=y_i\;\text{for}\;i\in\mathcal D_L.
\]

Known and novel prototype sets are coarsened separately at later levels.
Pseudo-label SupCon is accumulated with geometric level decay:

\[
\mathcal L_{hier}=\sum_{h=1}^{H}2^{-h}
\mathcal L_{SupCon}(z^1,z^2,q^{(h)}).
\]

This preserves SelEx's hierarchical self-expertise principle but does not
claim exact reproduction of its confusion smoothing or feature slicing.

## GET-adapted dual branches and CICO

Visual features and reliability-weighted attribute summaries share cluster
prototypes but produce separate logits. Both receive supervised and cross-view
objectives. For labeled classes, modality-specific anchors are

\[
\mu_{v,c}=|B_c|^{-1}\sum_{i\in B_c}v_i,\qquad
\mu_{t,c}=|B_c|^{-1}\sum_{i\in B_c}\bar t_i.
\]

CICO exchanges detached anchor-distribution targets:

\[
\mathcal L_{CICO}=H(\operatorname{sg}(p_t),p_v)
+H(\operatorname{sg}(p_v),p_t).
\]

Unlike GET, text here is observed OCR/attribute evidence rather than generated
TES pseudo text. This is a task-motivated adaptation.

## DebGCD-adapted semantic distribution and debiased heads

For every known category, an OVA head predicts `[unknown, known]`. Labeled SDL
rewards the true positive and hardest negative; unlabeled SDL combines OVA
entropy minimization and cross-view consistency. Distribution certainty is

\[
s_i=|2p_{unk}(i,\hat c_i)-1|.
\]

A separate debiased classifier receives labeled CE plus confidence-filtered
pseudo-label CE weighted by this certainty.

## Training objectives

### Supervised classification

Only rows explicitly marked `is_labeled=1` contribute:

\[
\mathcal{L}_{sup}=\frac{1}{|B_L|}\sum_{i\in B_L}
\operatorname{CE}(\ell_i,y_i).
\]

### Cross-view pseudo-label consistency

Let \(p_i^q=\operatorname{softmax}(\ell_i^q)\). A sample is accepted when both
views exceed epoch-dependent threshold \(\delta_e\). The symmetric loss is

\[
\mathcal{L}_{cons}=\frac{1}{2}\left[
D_{KL}(\operatorname{sg}(p^2)\Vert p^1)+
D_{KL}(\operatorname{sg}(p^1)\Vert p^2)
\right].
\]

The threshold increases linearly from zero to the configured target during
warm-up, preventing all novel samples from being rejected at initialization.

### Instance contrastive learning

For the two augmented views, NT-Xent is

\[
\mathcal{L}_{inst}=-\frac{1}{2B}\sum_i
\log\frac{\exp(z_i^1\cdot z_i^2/\tau_c)}
{\sum_{k\ne i}\exp(z_i^1\cdot z_k/\tau_c)}.
\]

### Supervised contrastive learning

For labeled anchors, all views sharing the same known label form the positive
set. Unlabeled ground-truth labels never enter this term.

### Anti-collapse regularization

For batch-mean cluster probability \(\bar p\),

\[
\mathcal{L}_{ent}=\sum_k\bar p_k\log(\bar p_k).
\]

Minimizing this negative entropy discourages assigning every sample to one
cluster.

### Reliability calibration

The current soft calibration target combines upstream confidence and detached
cross-modal agreement:

\[
\hat r_i^m=\sqrt{c_i^m\frac{1+\cos(v_i,t_i^m)}{2}},\quad
\mathcal{L}_{cal}=\operatorname{BCE}(r_i^m,\hat r_i^m).
\]

This is a modeling assumption and must be ablated. If attribute-correctness
annotations are available, they should replace the soft target.

### Conflict-aware pair constraints

For pairs with at least two matching reliable attributes and no conflict, a
must-link margin is applied. A disagreement on any reliable identity-bearing
attribute creates a cannot-link constraint:

\[
\mathcal{L}_{pair}=
\mathbb{E}_{(i,j)\in\mathcal{P}}[m_+-z_i^\top z_j]_+
+\mathbb{E}_{(i,j)\in\mathcal{C}}[z_i^\top z_j-m_-]_+.
\]

The current implementation treats all configured attributes as identity
bearing. A dataset-specific release should explicitly configure which fields
are allowed to create cannot-links.

### Total objective

\[
\mathcal{L}=\lambda_s\mathcal{L}_{sup}
+\lambda_c\mathcal{L}_{cons}
+\lambda_d\mathcal{L}_{distill}
+\lambda_i\mathcal{L}_{inst}
+\lambda_{sc}\mathcal{L}_{supcon}
+\lambda_h\mathcal{L}_{hier}
+\lambda_g\mathcal{L}_{dual}
+\lambda_{cico}\mathcal{L}_{CICO}
+\lambda_{sdl}\mathcal{L}_{SDL}
+\lambda_{adl}\mathcal{L}_{ADL}
+\lambda_e\mathcal{L}_{ent}
+\lambda_a\mathcal{L}_{must}
+\lambda_x\mathcal{L}_{cannot}
+\lambda_r\mathcal{L}_{cal}.
\]

All coefficients are explicit in JSON configs and are copied into every run
directory.

## Evaluation

One Hungarian assignment is fitted globally and reused for All, Old, and New
accuracy. Fitting separate mappings on subsets is prohibited because one
predicted cluster could otherwise map to different ground-truth classes.

The primary selection metric is the harmonic mean

\[
H=\frac{2A_{old}A_{new}}{A_{old}+A_{new}}.
\]

NMI, ARI, macro accuracy, head/medium/tail accuracy, and sibling-SKU accuracy
are also reported. Every table must state whether total class count is oracle
or estimated.

