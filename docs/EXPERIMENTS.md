# Experiment matrix

## Main comparison

Keep the manifest, class split, CLIP checkpoint, augmentation, optimizer, and
training budget fixed across all methods.

| Run | Fusion | Enabled paper-specific mechanism |
|---|---|---|
| `simgcd_style` | visual | Shared GCD base only |
| `selex_adapted` | visual | Hierarchical semi-supervised self-expertise |
| `get_adapted` | simple | Observed-attribute dual branches and CICO |
| `debgcd_adapted` | visual | Semantic distribution and auxiliary debiased learning |
| `simple_fusion` | simple | Equal visual/attribute fusion control |
| `proposed` | reliability-aware | All adapted mechanisms plus attribute reliability/conflicts |

Use seeds 1, 2, and 3 at minimum. Report mean and standard deviation. Names
ending in `_adapted` are not official reproductions and must not be presented
as such.

## Required ablations

1. `lambda_calibration=0`
2. `lambda_conflict=0`
3. `lambda_attribute=0`
4. `lambda_instance_contrastive=0`
5. `lambda_supervised_contrastive=0`
6. `lambda_hierarchy=0`
7. `lambda_cico=0`
8. `lambda_sdl=0` and `lambda_adl=0`
9. reliability replaced by raw OCR confidence (`simple_fusion`)
10. natural OCR versus leakage-controlled OCR
11. 10%, 30%, and 50% attribute corruption
12. random attribute missingness
13. balanced versus long-tailed subsets
14. oracle-K versus estimated-K
15. catalog-to-catalog, shelf-to-shelf, and catalog-to-shelf protocols

## Official-repository comparisons

Adapted baselines cannot replace official paper reproductions in a comparison
table. If official SelEx, GET, or DebGCD numbers are included, run their exact
commits separately under a matched retail dataset/backbone protocol and label
them as official-port results. Differences in TES, DINO versus CLIP, class
count, and dataset splits must be disclosed.

## Go/no-go criterion

The reliability claim is supported only if `proposed` improves over
`simple_fusion`, `get_adapted`, and `debgcd_adapted` on sibling-SKU and
corrupted-OCR subsets across multiple seeds—not merely aggregate accuracy from
one split.
