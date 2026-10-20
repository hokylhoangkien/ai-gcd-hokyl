# Dataset và hướng dẫn train cho người nhận GPU

## Trạng thái workspace hiện tại

Workspace hiện có GroceryStoreDataset đã được kiểm tra đầy đủ tại
`data/raw/GroceryStoreDataset/dataset` và manifest sạch tại
`data/processed/grocery_store_manifest.csv`. Manifest có 5.421 ảnh, 40 lớp known,
41 lớp novel, không có exact duplicate trong manifest và đã loại các attribute trùng
thẳng `sku_id` để tránh label leakage.

Chạy kiểm tra nhanh, không tải pretrained weights và không dùng kết quả này trong paper:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\retail-gcd.exe --config configs/grocery_store_preflight.json
```

Chạy benchmark GroceryStore đầy đủ trên máy GPU:

```powershell
.\.venv\Scripts\retail-gcd.exe --config configs/grocery_store_proposed.json
```

`configs/grocery_store_proposed.json` dùng pretrained CLIP; lần chạy đầu cần mạng để tải
weights. GroceryStore chỉ là public benchmark bổ sung. Claim chính của nghiên cứu vẫn cần
planogram crops như phần tiếp theo mô tả.

Thư mục `.restore_grocery` là bản sao tạm đầy đủ của `data/raw/GroceryStoreDataset` và
không được code tham chiếu. Có thể xóa để tránh nhầm dataset và tiết kiệm khoảng 123 MB.

## Chốt dataset nào

### Dataset chính bắt buộc

Hãy dùng **ảnh crop từng SKU từ dữ liệu shelf/planogram của RetailVision**. Đây
là dataset duy nhất phù hợp để bảo vệ claim chính của paper vì nó có đúng các
điều kiện cần nghiên cứu:

- sản phẩm được chụp trên kệ thật;
- nhiều SKU cùng brand/product line nhìn gần giống nhau;
- có ảnh mờ, nghiêng, che khuất và thay đổi ánh sáng;
- có text trên bao bì để chạy OCR;
- có SKU ID ground truth để đánh giá known/novel discovery.

Không train paper chính bằng GroceryStoreDataset. Bộ đó chủ yếu là rau quả và
thuộc tính semantic sạch, không đại diện bài toán packaged-SKU planogram.

### Public dataset bổ sung

Nếu cần một benchmark công khai, ưu tiên theo thứ tự:

1. **RP2K**: phù hợp nhất để kiểm tra fine-grained retail product recognition ở
   quy mô lớn. Chuyển class sản phẩm thành known/novel theo cùng protocol.
2. **RPC (Retail Product Checkout)**: phù hợp làm kiểm tra catalog-to-real hoặc
   domain shift, nhưng bối cảnh checkout không hoàn toàn giống shelf planogram.
3. **Products-10K**: có thể dùng làm visual pretraining/benchmark bổ sung, không
   thay thế shelf test set.
4. **SKU-110K**: chỉ dùng cho detector/cropper vì dataset tập trung bounding box
   trong kệ dày đặc; không dùng làm benchmark SKU-level GCD chính.

Public dataset không có sẵn các cột OCR/attribute giống RetailVision. Nếu dùng,
phải chạy cùng OCR pipeline trên ảnh, hoặc chỉ báo kết quả visual-only. Không tự
điền class name vào attribute của ảnh novel vì đó là label leakage.

## Dữ liệu người train cần nhận

```text
data/
  processed/
    planogram_crops/
      store01/session01/crop_000001.jpg
      store01/session01/crop_000002.jpg
      ...
    crop_annotations.csv
```

Mỗi ảnh phải là một crop chứa chủ yếu một sản phẩm. `crop_annotations.csv` cần
ít nhất:

```csv
image_path,sku_id,source_split,is_known,session_id,source,brand,brand_conf,product_line,product_line_conf,variant,variant_conf,volume,volume_conf,pack_count,pack_count_conf,packaging,packaging_conf
store01/session01/crop_000001.jpg,sku_001,train,1,store01_session01,shelf,Coca-Cola,0.97,Coca-Cola,0.91,Zero,0.82,330 ml,0.76,,0.0,can,0.94
```

Quy tắc:

- `image_path` tương đối so với `data/processed/planogram_crops`;
- `sku_id` là ground truth dùng để split/evaluate, không được đưa trực tiếp vào
  encoder;
- `source_split` là `train`, `val` hoặc `test`;
- `is_known` cố định cho toàn bộ ảnh của một SKU;
- `session_id` là store/camera/video burst; một session chỉ thuộc một split;
- confidence thuộc `[0,1]`; thiếu giá trị thì để rỗng và confidence bằng `0`;
- xóa barcode, database SKU ID và full catalog name ở leakage-controlled run.

## Chia GCD đúng cách

Ví dụ có 200 SKU:

- 100 SKU known, 100 SKU novel;
- known train: một phần ảnh vào `labeled`, phần còn lại vẫn vào `unlabeled`;
- novel train: toàn bộ chỉ vào `unlabeled`, tuyệt đối không có labeled sample;
- `val` và `test` đều chứa ảnh của known lẫn novel SKU;
- mỗi novel SKU cần có ảnh trong train-unlabeled để mô hình khám phá, sau đó mới
  được đánh giá trên val/test;
- chia ảnh theo store/session/video burst, không random frame riêng lẻ;
- loại exact duplicate và near-duplicate giữa train/val/test.

Mức dữ liệu thực dụng nên nhắm tới:

- ít nhất 100–200 SKU để pilot có ý nghĩa;
- lý tưởng 20–50 crop trở lên cho mỗi SKU;
- nhiều store, camera, ngày và góc chụp;
- không để một SKU chỉ có đúng một bối cảnh nền.

Đây là mục tiêu thu thập, không phải điều kiện cứng của code.

## Chuẩn bị manifest

Từ root repository:

```bash
python scripts/prepare_planogram_manifest.py \
  --annotations data/processed/crop_annotations.csv \
  --output data/processed/planogram_manifest.csv \
  --known-ratio 0.5 \
  --labeled-ratio 0.5 \
  --seed 1
```

Nếu CSV đã có `is_known`, script giữ class split đó; `--known-ratio` chỉ được
dùng khi chưa cung cấp `is_known`.

Kiểm tra manifest và đường dẫn ảnh:

```bash
retail-gcd-validate data/processed/planogram_manifest.csv \
  --image-root data/processed/planogram_crops
```

Không train nếu validator báo lỗi, thiếu `val`, thiếu known/novel ở evaluation,
hoặc có duplicate xuyên split.

## Cài môi trường trên máy GPU

Khuyến nghị Linux, Python 3.10/3.11 và một NVIDIA GPU từ 16 GB VRAM. Repository
hiện chạy một GPU; chưa có DDP/gradient accumulation trong training path.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Chọn đúng CUDA wheel tại pytorch.org; ví dụ dưới đây chỉ là mẫu.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
python -m pytest -q
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Không chạy dài nếu `torch.cuda.is_available()` là `False`.

## Cấu hình

Mở `configs/planogram_proposed.json` và kiểm tra:

```json
"manifest": "data/processed/planogram_manifest.csv",
"image_root": "data/processed/planogram_crops",
"batch_size": 32,
"num_workers": 8
```

Nếu out-of-memory, giảm batch size xuống 16 hoặc 8. Không đổi backbone, split,
augment hoặc training budget giữa các baseline.

Sau khi sửa base config, luôn sinh lại ma trận:

```bash
python scripts/generate_experiments.py --seeds 1 2 3
```

## Thứ tự chạy

### Bước 1: preflight

Copy `configs/planogram_proposed.json` thành một config tạm và đặt:

```json
"epochs": 1,
"max_train_batches": 2,
"max_eval_batches": 2
```

Chạy:

```bash
retail-gcd --config configs/planogram_preflight.json
```

Xác nhận loss hữu hạn, GPU memory ổn định, `last.pt`/`best.pt` được tạo và val
có cả known lẫn novel. Xóa ba giới hạn preflight khỏi config dùng cho kết quả.

### Bước 2: pilot một seed

Chạy seed 1 theo thứ tự:

```bash
retail-gcd --config configs/generated/simgcd_style_seed1.json
retail-gcd --config configs/generated/simple_fusion_seed1.json
retail-gcd --config configs/generated/selex_adapted_seed1.json
retail-gcd --config configs/generated/get_adapted_seed1.json
retail-gcd --config configs/generated/debgcd_adapted_seed1.json
retail-gcd --config configs/generated/proposed_seed1.json
```

Nếu proposed không hơn baseline nào trên val, dừng để phân tích lỗi; không đốt
GPU chạy thêm seed ngay.

### Bước 3: full ba seed

Khi pilot hợp lý, chạy toàn bộ lệnh trong `configs/generated/commands.txt`. Mỗi
run có output directory riêng. Không ghi đè seed.

### Bước 4: test offline

Checkpoint được chọn trên val trong quá trình train. Sau khi khóa mọi lựa chọn,
đánh giá test:

```bash
retail-gcd-eval \
  --checkpoint outputs/planogram/proposed/seed_1/best.pt \
  --splits test \
  --head primary
```

Lặp lại cho ba seed và các baseline. Không thêm `test` vào
`data.validation_splits`; config validator sẽ từ chối.

## Kết quả người train phải trả lại

Gửi lại toàn bộ `outputs/planogram/`, cùng một bản copy manifest đã dùng và log
môi trường. Bảng tổng hợp phải có:

- All, Old/Known, New/Novel và harmonic accuracy;
- NMI, ARI, mean ± standard deviation qua ba seed;
- sibling-SKU accuracy;
- head/medium/tail classes;
- OCR missing/corruption robustness;
- natural-OCR và leakage-controlled OCR;
- oracle-K và estimated-K ghi thành hai protocol khác nhau;
- cấu hình, commit code, GPU và phiên bản PyTorch/OpenCLIP.

Không dùng metric từ preflight/smoke test trong paper.
