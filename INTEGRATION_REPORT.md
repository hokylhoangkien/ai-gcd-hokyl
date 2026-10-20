# Báo cáo tích hợp cuối: Retail GCD

Ngày kiểm định: 2026-10-20

## Kết luận trung thực

Repository hiện đã có một pipeline huấn luyện GCD hợp nhất, chạy được từ đầu đến
cuối và có các nhánh thích nghi từ SelEx, GET và DebGCD. Tuy nhiên, **không thể
bảo đảm trước rằng kết quả sẽ vượt từng paper** khi chưa huấn luyện trên dữ liệu
planogram thật, chưa chạy nhiều seed và chưa so sánh cùng protocol. Smoke test
chỉ chứng minh code path hoạt động; nó không phải bằng chứng SOTA.

Mức tuyên bố đúng ở thời điểm này là:

- code hợp nhất đã được hiện thực và kiểm tra;
- nguồn ý tưởng, commit upstream và khác biệt đều được ghi lại;
- protocol đã ngăn test leakage khi chọn checkpoint;
- 18 cấu hình đối chứng (6 phương pháp × 3 seed) đã sẵn sàng;
- việc mô hình có “wow” hay không phải được quyết định bằng kết quả GPU.

## Mô hình cuối cùng

Tên làm việc:

> Reliability-Aware Attribute-Guided Generalized Category Discovery for
> Fine-Grained Retail SKU Recognition

Luồng chính:

```text
product crop
   ├── OpenCLIP visual encoder ──> visual feature ──> visual logits
   └── OCR/structured attributes ─> attribute features ─> attribute logits
                                      │
visual/attribute agreement + OCR confidence
                                      │
                         per-attribute reliability
                                      │
              reliability-weighted multimodal fusion
                                      │
                    primary GCD cluster logits
                         ├── hierarchy objective
                         ├── OVA semantic-distribution head
                         ├── auxiliary debiased head
                         └── attribute must-link/cannot-link constraints
```

Đầu vào của GCD là **crop của từng sản phẩm**, không phải toàn bộ ảnh kệ. Detector
và kiểm tra planogram là tầng trước/sau; repository này tập trung vào nhận diện
known SKU và khám phá novel SKU trong các crop.

## Đã lấy ý tưởng gì từ đâu và ghép như thế nào

Không có file Python upstream nào được copy nguyên vào `src/retail_gcd/`. Code
production là clean local implementation dựa trên cơ chế đã đọc từ các commit
cố định dưới đây.

| Nguồn | Commit đã đọc | Cơ chế được thích nghi | Code local | Khác với bản gốc |
|---|---|---|---|---|
| SelEx, ECCV 2024 | `569ee7085e779999502bd73ea92240f3d32fc84d` | mixed semi-supervised K-means, progressive known/novel hierarchy, self-expertise | `src/retail_gcd/evaluation/hierarchy.py`, `hierarchical_contrastive_loss` và `refresh_hierarchy` | dùng OpenCLIP; chưa tái tạo đúng confusion-factor smoothing và feature slicing của SelEx |
| GET, CVPR 2025 | `8e908c061660ce95b07596b9a7f190e5c444f7cb` | hai nhánh visual/text, branch losses, CICO anchor-distribution consistency | `src/retail_gcd/model.py`, `cross_modal_anchor_consistency`, `train_epoch` | thay TES pseudo-text bằng OCR/thuộc tính quan sát được từ bao bì |
| DebGCD, ICLR 2025 | `9ec30fe3d54dfccd9934a862cf5cb9ff05de6f05` | OVA semantic distribution learning, certainty, auxiliary debiased classifier | distribution/debiased heads trong `model.py`; SDL/ADL trong `losses.py` | dùng backbone và format retail riêng; không phải official reproduction |
| SimGCD/DINO/SimCLR/SupCon family | cơ chế chuẩn trong GCD | two-view learning, self-distillation, InfoNCE, SupCon, entropy anti-collapse | `src/retail_gcd/train.py`, `src/retail_gcd/losses.py` | local implementation với manifest và schedule riêng |
| Đề xuất của repository | local | reliability theo từng thuộc tính và conflict-aware constraints | reliability/fusion trong `model.py`; calibration, must-link/cannot-link trong `losses.py` | đóng góp cần được ablation và literature review xác nhận |

Danh sách file upstream đã đọc, số file/dòng và đường đi cơ chế chi tiết nằm ở
`UPSTREAM_AUDIT.md`. Quy tắc trích dẫn và provenance nằm ở `PROVENANCE.md`.

### SelEx-adapted

`build_hierarchy` chạy semi-supervised K-means trên toàn bộ train set. Assignment
của mẫu labeled bị khóa theo nhãn known; novel assignments được khám phá. Các
mức hierarchy coarsen known và novel riêng, tránh nhập nhằng ID giữa hai nhóm.
Pseudo-label ở từng mức tạo supervised-contrastive objective có trọng số giảm
dần. Hierarchy được refresh định kỳ, sau giai đoạn warm-up.

Đây là **SelEx-adapted**, không được ghi là kết quả official SelEx.

### GET-adapted

Mô hình xuất đồng thời `visual_logits`, `attribute_logits` và fused `logits`.
Mỗi nhánh nhận supervised/cross-view loss. CICO so khớp phân phối của ảnh và
thuộc tính đối với anchor của known classes theo hai chiều.

GET gốc tổng hợp pseudo-word từ ảnh bằng TES. Bài toán này đã có OCR/brand/
variant/volume quan sát được, nên dùng các thuộc tính đó thay vì biến lại thông
tin ảnh thành pseudo-text. Đây là **GET-adapted**.

### DebGCD-adapted

Ngoài primary head, mô hình có:

- OVA distribution head: với mỗi known class, ước lượng quan hệ known/unknown;
- debiased head: học labeled CE và pseudo-label tin cậy;
- distribution certainty: cân trọng số pseudo-label cho auxiliary head.

Known class luôn chiếm chỉ số `[0, K_known)` để đúng layout OVA. Đây là
**DebGCD-adapted**, không phải reproduction chính thức.

### Phần đề xuất riêng

Mỗi thuộc tính có reliability riêng, không dùng một scalar chung cho cả nhánh
text. Reliability dùng OCR confidence, trạng thái missing và mức đồng thuận
visual–attribute để điều khiển fusion. Cặp có nhiều thuộc tính đáng tin cậy giống
nhau tạo must-link; cặp khác ở `variant`, `volume`, `pack_count`, v.v. tạo
cannot-link. Thuộc tính yếu không được ép constraint.

Đây là phần có khả năng tạo giá trị riêng cho planogram/SKU, nhưng chỉ có thể
gọi là đóng góp mới sau khi ablation chứng minh hiệu quả.

## Objective và lịch huấn luyện

Objective tổng gồm:

```text
L = L_supervised + L_consistency + L_distillation
  + L_InfoNCE + L_SupCon + L_entropy
  + L_dual_branch + L_CICO
  + L_SDL + L_ADL + L_hierarchy
  + L_attribute_must_link + L_conflict_cannot_link + L_calibration
```

Không bật mạnh mọi module từ epoch đầu. `planogram_proposed.json` dùng start/
warm-up riêng cho CICO, SDL, ADL, hierarchy và conflict. Điều này giảm nguy cơ
pseudo-label nhiễu ở đầu quá trình tự khuếch đại sai số.

## Những lỗi protocol đã khóa trong lần kiểm định cuối

1. Checkpoint chỉ được chọn trên `data.validation_splits` và cấu hình có `test`
   ở đây sẽ bị từ chối bằng `ConfigurationError`.
2. Test chỉ được chạy offline bằng `retail-gcd-eval` sau khi checkpoint đã chốt.
3. Horizontal flip mặc định tắt vì lật chữ/bao bì tạo ảnh sản phẩm phi thực tế.
4. Train sampler cân bằng labeled/unlabeled để nhóm labeled nhỏ không biến mất.
5. Known IDs được xếp liên tục trước novel IDs cho OVA head.
6. Novel target vẫn tồn tại để đánh giá nhưng mọi supervised loss đều mask bằng
   `is_labeled`.
7. Metric All/Old/New dùng cùng một global Hungarian assignment, tránh cùng một
   cluster được remap khác nhau ở từng subset.
8. Checkpoint, config, class mapping, seed, scheduler và optimizer state được lưu
   để có thể resume và audit.

## Ma trận thí nghiệm đã tạo

`scripts/generate_experiments.py` tạo 18 cấu hình trong `configs/generated/`:

| Cấu hình | Mục đích |
|---|---|
| `simgcd_style` | visual-only GCD base |
| `selex_adapted` | cô lập hierarchy/self-expertise |
| `get_adapted` | cô lập dual branch + CICO |
| `debgcd_adapted` | cô lập SDL + ADL |
| `simple_fusion` | kiểm tra lợi ích chỉ do thêm attribute |
| `proposed` | toàn bộ cơ chế thích nghi + reliability/conflict |

Mỗi cấu hình chạy seed 1, 2 và 3. Các local baseline `_adapted` không thay thế
official paper reproduction. Nếu paper tuyên bố vượt SelEx/GET/DebGCD, phải chạy
code chính thức cùng dataset split, backbone/class-count protocol hoặc ghi rõ số
từ paper không trực tiếp so sánh được.

## Điều kiện mới được phép nói “tốt hơn”

Không chọn kết luận từ một seed hoặc một chỉ số. Proposed chỉ được coi là thắng
thuyết phục khi:

- mean ± std qua ít nhất 3 seed tốt hơn baseline mạnh nhất;
- tăng `new_acc` và harmonic mean mà không làm giảm mạnh `old_acc`;
- tăng rõ trên sibling-SKU subset, không chỉ aggregate accuracy;
- bền hơn khi OCR bị corrupt/missing và dưới long-tail imbalance;
- ablation cho thấy bỏ reliability hoặc conflict làm kết quả giảm;
- gain không biến mất khi cùng backbone, augment, K và split;
- test set chỉ được mở sau khi hyperparameter/checkpoint đã khóa;
- nên báo bootstrap confidence interval hoặc paired significance test.

Một ngưỡng thực dụng để quyết định GO là improvement ổn định khoảng 2 điểm phần
trăm trở lên trên novel/sibling-SKU harmonic accuracy với variance nhỏ và không
đổi dấu giữa các seed. Đây là tiêu chí dự án, không phải bảo đảm trước kết quả.

## Trạng thái kiểm tra code

Lần kiểm định 2026-10-20:

- sinh lại đủ 18 config: pass;
- `pytest`: **9/9 pass**;
- compile toàn bộ `src`, `scripts`, `tests`: pass;
- CPU end-to-end smoke run: pass;
- smoke run đã đi qua forward, backward, optimizer, scheduler, val evaluation và
  atomic checkpoint writing;
- full multi-seed planogram training: chưa chạy vì máy hiện tại không có GPU và
  chưa có dữ liệu planogram thật trong repository.

Giá trị metric của one-batch smoke run không có ý nghĩa khoa học và không được
đưa vào paper.

## Lệnh giao cho máy GPU

```bash
python scripts/generate_experiments.py
python -m pytest -q
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Sửa đường dẫn manifest/image trong `configs/planogram_proposed.json`, sinh lại
config, rồi chạy tuần tự các lệnh trong:

```text
configs/generated/commands.txt
```

Sau khi chốt checkpoint trên val, test đúng một lần cho mỗi seed/head:

```bash
retail-gcd-eval \
  --checkpoint outputs/planogram/proposed/seed_1/best.pt \
  --splits test \
  --head primary
```

Hướng dẫn môi trường, preflight, K-estimation và danh sách kết quả cần báo nằm ở
`TRAINING_HANDOFF.md`. Công thức đầy đủ nằm ở `docs/METHOD.md`; ma trận ablation
nằm ở `docs/EXPERIMENTS.md`.

## Rủi ro còn lại

- Chưa có kết quả GPU trên dữ liệu planogram thật, nên chưa biết module nào tạo
  gain hay gây negative transfer.
- Số cluster chính hiện dùng oracle K; estimated-K là protocol offline riêng.
- SelEx-adapted chưa có confusion-factor smoothing y hệt upstream.
- GET-adapted cố ý không dùng TES.
- Không bundle GroceryStore trong bản repo sạch vì nó không đại diện packaged
  planogram SKU; paper chính phải dùng RetailVision/planogram crops.
- OCR có thể leak SKU nếu input chứa barcode/mã định danh đầy đủ; phải audit và
  báo riêng leakage-controlled protocol.

## Quyết định cuối

Code hiện đủ trưởng thành để **bắt đầu pilot GPU và ablation**, chưa đủ bằng
chứng để hứa SOTA. Cách nhanh nhất để đạt kết quả tốt không phải thêm tiếp module,
mà là chạy ma trận đã khóa, đọc failure cases, rồi chỉ giữ những thành phần có
gain ổn định trên planogram thật.
