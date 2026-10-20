# Đề xuất nghiên cứu: Reliability-Aware Attribute-Guided GCD for Retail SKU Discovery

## 1. Hướng nghiên cứu được chọn

**Tên đề tài đề xuất**

> **Reliability-Aware Attribute-Guided Generalized Category Discovery for Fine-Grained Retail SKU Recognition**

**Tên tiếng Việt**

> **Khám phá danh mục tổng quát dựa trên độ tin cậy của thuộc tính cho nhận diện SKU bán lẻ fine-grained**

Đây là hướng nghiên cứu chính thức được chọn. Trọng tâm không phải là ghép nguyên khối nhiều phương pháp GCD hiện có, mà là giải quyết một hạn chế cụ thể của nhận diện SKU ngoài thực tế: các SKU gần giống nhau thường chỉ khác một vài thuộc tính nhỏ, trong khi thông tin hình ảnh và chữ trên bao bì có thể bị thiếu, sai hoặc mâu thuẫn.

---

## 2. Bài toán nghiên cứu

Trong hệ thống retail, một detector trước tiên phát hiện và cắt từng sản phẩm từ ảnh kệ hàng. Với các product crop này, hệ thống phải đồng thời:

1. Nhận diện những SKU đã biết và đã có nhãn.
2. Phát hiện các SKU chưa từng được gán nhãn.
3. Gom ảnh của các SKU mới thành những cụm đúng.
4. Không gộp nhầm các SKU rất giống nhau nhưng khác variant, hương vị, dung tích hoặc pack count.

Thiết lập dữ liệu GCD gồm:

- **Labeled set:** chỉ chứa một phần các lớp SKU đã biết.
- **Unlabeled set:** chứa cả SKU đã biết và SKU mới.
- **Novel classes:** không có nhãn trong quá trình huấn luyện.
- **Đầu ra:** dự đoán đúng lớp cũ và khám phá các cụm SKU mới.

Ví dụ khó:

```text
Coca-Cola Zero 330 ml
Coca-Cola Zero 500 ml
Coca-Cola Original 330 ml
```

Các sản phẩm này có phần lớn đặc trưng hình ảnh và từ ngữ giống nhau. Tín hiệu phân biệt có thể chỉ là một vùng nhỏ hoặc một thuộc tính như `variant` hay `volume`.

---

## 3. Khoảng trống nghiên cứu

Các phương pháp GCD thị giác có thể không nhận ra khác biệt nhỏ giữa những SKU cùng họ sản phẩm. Phương pháp đa phương thức có thể tận dụng text, nhưng fusion thông thường vẫn gặp ba vấn đề:

1. **Độ tin cậy thay đổi theo từng ảnh:** OCR tốt ở ảnh này nhưng sai ở ảnh khác.
2. **Độ tin cậy thay đổi theo từng thuộc tính:** OCR có thể đọc đúng brand nhưng sai dung tích.
3. **Xung đột giữa các modality:** hình ảnh cho thấy hai sản phẩm rất giống nhau, trong khi thuộc tính định danh cho thấy chúng là hai SKU khác nhau.

Vì vậy, chỉ thêm OCR hoặc một text encoder chưa giải quyết đầy đủ bài toán. Khoảng trống cần nghiên cứu là cách sử dụng **độ tin cậy ở cấp thuộc tính** để tạo pseudo-label và cụm novel class chính xác hơn.

---

## 4. Câu hỏi và giả thuyết nghiên cứu

### Câu hỏi nghiên cứu

> Can attribute-level reliability and conflict-aware pseudo-labeling improve fine-grained generalized category discovery when visual and textual evidence is incomplete, noisy, or contradictory?

### Giả thuyết

> Việc ước lượng độ tin cậy riêng cho các thuộc tính định danh như brand, product line, variant, volume và pack count sẽ tạo pseudo-label sạch hơn, qua đó hạn chế việc gộp nhầm các sibling SKU vào cùng một novel cluster.

---

## 5. Phương pháp đề xuất

### 5.1. Kiến trúc tổng quát

```text
Product crop
   |
   +-- Visual encoder (DINOv2 hoặc CLIP)
   |
   +-- OCR + text encoder
   |
   +-- Attribute parser
         +-- brand
         +-- product line
         +-- variant/flavor
         +-- size/volume
         +-- pack count
         +-- packaging type
   |
   v
Attribute-level reliability estimation
   |
   v
Conflict-aware affinity graph
   |
   v
Reliability-aware pseudo-labeling
   |
   v
Known-SKU classification + Novel-SKU clustering
```

### 5.2. Visual representation

Dùng một backbone pretrained mạnh như DINOv2 hoặc CLIP để trích xuất:

- Hình dạng và loại bao bì.
- Màu sắc và thiết kế.
- Logo và vùng nhận diện thương hiệu.
- Các chi tiết fine-grained trên nhãn.

Backbone phải được giữ cố định hoặc cấu hình giống nhau giữa các phương pháp khi so sánh.

### 5.3. Text và thuộc tính có cấu trúc

OCR được sử dụng để đọc chữ xuất hiện thật trên product crop. Text sau đó được chuyển thành các thuộc tính có cấu trúc:

```json
{
  "brand": "Coca-Cola",
  "product_line": "Coca-Cola",
  "variant": "Zero",
  "volume": "330 ml",
  "pack_count": null,
  "packaging": "can"
}
```

Không giả định mọi ảnh đều có đầy đủ thuộc tính. Mỗi thuộc tính có thể đúng, sai hoặc bị thiếu.

### 5.4. Attribute-level reliability

Với mỗi ảnh, mô hình dự đoán một độ tin cậy riêng cho từng thuộc tính:

```text
r_brand, r_product, r_variant, r_size, r_pack, r_packaging
```

Reliability có thể dựa trên:

- Confidence và độ ổn định của OCR.
- Chất lượng vùng ảnh chứa text.
- Mức đồng thuận giữa visual representation và text.
- Tính nhất quán giữa nhiều augmentation hoặc nhiều ảnh lân cận.
- Mức hợp lệ của giá trị thuộc tính, ví dụ định dạng dung tích.

Điểm mới cần bảo vệ là reliability ở **cấp thuộc tính**, không phải một scalar chung cho toàn bộ nhánh text.

### 5.5. Conflict-aware affinity graph

Mỗi ảnh là một node. Trọng số cạnh giữa hai ảnh được tính từ:

- Visual similarity.
- Text/attribute similarity.
- Reliability của từng thuộc tính.
- Mức xung đột trên thuộc tính định danh.

Nguyên tắc:

- Visual giống và thuộc tính đáng tin cậy giống nhau: positive edge mạnh.
- Visual giống nhưng `volume` hoặc `variant` đáng tin cậy khác nhau: giảm mạnh affinity hoặc tạo cannot-link constraint.
- OCR kém tin cậy: không dùng text để áp đặt cannot-link.
- Text giống nhưng visual rất khác: giữ confidence thấp cho đến khi có thêm bằng chứng.

### 5.6. Reliability-aware pseudo-labeling

Pseudo-label chỉ được dùng mạnh khi:

- Quan hệ ảnh–ảnh đủ chắc chắn.
- Các modality không có xung đột đáng tin cậy.
- Dự đoán ổn định qua augmentation hoặc các vòng cập nhật.

Loss tổng quát có thể gồm:

```text
L = L_supervised
  + lambda_u * L_unsupervised
  + lambda_a * L_attribute_consistency
  + lambda_c * L_conflict_constraint
  + lambda_r * L_reliability_calibration
```

Chi tiết loss sẽ được quyết định sau khi baseline và error analysis xác nhận dạng lỗi thực tế. Không nên cố định kiến trúc phức tạp trước bước này.

---

## 6. Đóng góp dự kiến của paper

Paper hướng đến bốn đóng góp:

1. **Một formulation cho fine-grained retail GCD** trong điều kiện tín hiệu thị giác và text bị thiếu, nhiễu hoặc mâu thuẫn.
2. **Cơ chế attribute-level reliability** thay vì fusion với một confidence chung cho cả modality.
3. **Conflict-aware pseudo-labeling hoặc affinity learning** nhằm tránh gộp nhầm sibling SKU.
4. **Một protocol đánh giá retail GCD** bao gồm sibling SKU, OCR corruption, class imbalance và catalog-to-shelf domain shift.

Đóng góp thứ tư chỉ được tuyên bố là benchmark contribution nếu dữ liệu và protocol đủ quy mô, có thể công bố và được xây dựng nghiêm ngặt.

---

## 7. Dữ liệu và protocol

### 7.1. Loại dữ liệu cần có

- Product crops từ ảnh catalog và/hoặc ảnh kệ hàng.
- Ground-truth SKU chỉ dùng để tạo split và đánh giá.
- Text nhìn thấy trên bao bì hoặc OCR output.
- Metadata thuộc tính nếu có, dùng để đánh giá attribute extraction hoặc làm oracle experiment.
- Nhóm sibling SKU: cùng brand/product line nhưng khác variant, size hoặc pack count.

### 7.2. Cách chia dữ liệu

Phải split theo SKU và nguồn ảnh, không random từng ảnh một cách đơn giản.

- Một phần SKU được chọn làm known classes.
- Novel SKU không xuất hiện trong labeled set.
- Unlabeled set chứa cả known và novel SKU.
- Ảnh gần trùng hoặc frame liên tiếp không được nằm ở nhiều split.
- Nếu có nhiều cửa hàng hoặc phiên chụp, ưu tiên tách theo store/session.
- Chạy tối thiểu ba random seeds.

### 7.3. Ba protocol chính

1. **Catalog → Catalog:** kiểm tra bài toán trong điều kiện ảnh sạch.
2. **Shelf → Shelf:** kiểm tra trên ảnh thực tế có che khuất, nghiêng và ánh sáng xấu.
3. **Catalog → Shelf:** kiểm tra domain shift và khả năng áp dụng thực tế.

### 7.4. Kiểm soát label leakage

Cần báo cáo hai chế độ:

- **Natural OCR:** giữ lại toàn bộ text thực sự nhìn thấy trên bao bì.
- **Leakage-controlled OCR:** loại hoặc che barcode, SKU ID, tên định danh hoàn chỉnh và metadata không thể quan sát trực tiếp từ ảnh.

Không được lấy tên SKU đầy đủ từ catalog làm input rồi xem đó là OCR. Ground-truth metadata chỉ được dùng làm oracle hoặc target đánh giá, trừ khi protocol nêu rõ điều khác.

---

## 8. Baseline cần so sánh

### Baseline tối thiểu

1. Visual-only GCD với cùng backbone.
2. SelEx hoặc một baseline fine-grained GCD mạnh.
3. Text/OCR-only clustering.
4. Visual + text fusion đơn giản.
5. GET hoặc baseline multimodal GCD gần nhất có thể tái lập.
6. Oracle attributes để đo upper bound.
7. Phương pháp đề xuất.

### Baseline bổ sung

- DebGCD cho thí nghiệm mất cân bằng và bias.
- HypCD nếu cần đánh giá cấu trúc phân cấp.

DebGCD và HypCD không phải thành phần bắt buộc của mô hình chính. Chỉ tích hợp khi error analysis và ablation chứng minh chúng giải quyết một vấn đề còn tồn tại.

---

## 9. Chỉ số đánh giá

### Chỉ số GCD tiêu chuẩn

- All/Overall accuracy.
- Old/Known accuracy.
- New/Novel accuracy.
- Harmonic mean giữa Known và Novel.
- NMI.
- ARI.

### Chỉ số cần thiết cho retail

- Macro accuracy hoặc balanced accuracy.
- Head/medium/tail performance.
- Sibling-SKU accuracy.
- Novel class-count error khi số lớp mới không được cung cấp.
- Catalog-to-shelf performance drop.
- Kết quả theo mức OCR corruption hoặc missing attributes.
- Trung bình và độ lệch chuẩn qua ít nhất ba seeds.

### Đánh giá reliability

- Expected Calibration Error nếu phù hợp.
- Accuracy–confidence curve.
- Selective accuracy/coverage.
- Tương quan giữa reliability dự đoán và độ chính xác thật của từng thuộc tính.

---

## 10. Ablation study bắt buộc

1. Không sử dụng OCR/text.
2. Fusion trung bình hoặc concatenation đơn giản.
3. Một reliability score chung cho text.
4. Reliability riêng theo từng thuộc tính.
5. Bỏ conflict/cannot-link constraint.
6. Bỏ attribute consistency loss.
7. OCR corruption ở nhiều mức, ví dụ 10%, 30% và 50%.
8. Randomly missing attributes.
9. Balanced và long-tailed class distributions.
10. Biết và không biết trước số novel classes.

Ablation quan trọng nhất là chứng minh attribute-level reliability tốt hơn một confidence chung trong các nhóm sibling SKU.

---

## 11. Thí nghiệm xác nhận trước khi xây mô hình

Trước khi phát triển phương pháp mới, thực hiện một pilot trên khoảng 100–300 SKU, ưu tiên có 20–30 nhóm sibling SKU và khoảng 10–30 ảnh mỗi SKU nếu dữ liệu cho phép.

Chạy bốn hệ thống:

1. Visual-only.
2. OCR/text-only.
3. Simple visual–text fusion.
4. Oracle attributes.

### Tiêu chí ra quyết định

| Quan sát | Kết luận |
|---|---|
| Oracle attributes tốt hơn visual rõ rệt | Attribute-guided GCD có tiềm năng |
| OCR kém hơn oracle đáng kể | Có khoảng trống cho reliability/noise handling |
| Simple fusion đã gần oracle | Novelty reliability có thể không đủ mạnh |
| Text-only gần hoàn hảo | Có nguy cơ label leakage |
| Visual-only gần hoàn hảo | Dataset chưa đủ khó |
| Catalog tốt nhưng shelf giảm mạnh | Domain shift cần trở thành thành phần chính |

### Điều kiện GO

Tiếp tục hướng đã chọn khi đồng thời quan sát được:

- Sibling SKU là nguồn lỗi đáng kể.
- Oracle attributes cải thiện rõ so với visual-only.
- OCR thực tế nhiễu hơn đáng kể so với oracle.
- Simple fusion chưa giải quyết được xung đột giữa visual và attributes.

Nếu không đạt các điều kiện trên, không nên cố giữ phương pháp; cần chuyển trọng tâm sang domain-robust catalog-to-shelf GCD hoặc một vấn đề được dữ liệu thực tế hỗ trợ rõ hơn.

---

## 12. Lộ trình thực hiện

### Giai đoạn 1 — Xác nhận bài toán (2 tuần)

- Audit dữ liệu, duplicate và leakage.
- Xây known/novel splits và sibling groups.
- Chạy bốn pilot baselines.
- Phân tích định lượng và ít nhất 100 lỗi.
- Ra quyết định GO/NO-GO.

### Giai đoạn 2 — Chuẩn hóa baseline (3–4 tuần)

- Reproduce visual GCD baseline và SelEx.
- Reproduce multimodal baseline phù hợp.
- Cố định backbone, augmentation và evaluation protocol.
- Lưu config, seed và kết quả để có thể tái lập.

### Giai đoạn 3 — Phương pháp đề xuất (4–6 tuần)

- Xây attribute parser.
- Xây attribute-level reliability estimator.
- Xây conflict-aware affinity/pseudo-labeling.
- Bắt đầu bằng phiên bản nhỏ nhất có thể kiểm chứng.
- Chỉ thêm module nếu error analysis chứng minh cần thiết.

### Giai đoạn 4 — Kiểm chứng (3–4 tuần)

- Chạy full baselines và nhiều seeds.
- Ablation study.
- OCR corruption và missing attributes.
- Long-tail evaluation.
- Catalog-to-shelf evaluation.
- Qualitative cluster/retrieval visualization.

### Giai đoạn 5 — Viết paper (2–3 tuần)

- Viết problem formulation và related work.
- Tổng hợp bảng kết quả và kiểm định thống kê.
- Phân tích failure cases và limitations.
- Kiểm tra lại novelty với literature mới nhất trước khi submission.

---

## 13. Rủi ro và phương án xử lý

### Rủi ro 1: Phương pháp bị xem là ghép module

**Xử lý:** tập trung đóng góp vào formulation, attribute-level reliability và conflict-aware pseudo-labeling; so sánh trực tiếp với fusion/confidence thông thường.

### Rủi ro 2: OCR làm bài toán quá dễ

**Xử lý:** dùng leakage-controlled protocol và đánh giá dưới OCR corruption.

### Rủi ro 3: OCR quá kém để sử dụng

**Xử lý:** dùng missing-modality training; nếu oracle experiment cũng không cải thiện thì dừng hướng attribute-guided.

### Rủi ro 4: Không đủ dữ liệu shelf

**Xử lý:** dùng catalog cho nghiên cứu ban đầu, nhưng không đưa ra tuyên bố mạnh về triển khai thực tế; xây một test set shelf nhỏ có kiểm soát nếu có thể.

### Rủi ro 5: Cải thiện chỉ đến từ backbone

**Xử lý:** giữ cùng backbone và training budget cho mọi phương pháp so sánh.

### Rủi ro 6: Số novel classes được giả định biết trước

**Xử lý:** báo cáo cả thiết lập oracle class count và estimated class count.

---

## 14. Phạm vi không thực hiện ở phiên bản đầu

- Không copy hoặc ghép nguyên khối SelEx, DebGCD, GET và HypCD rồi gọi đó là đóng góp mới. Bản triển khai cuối có các cơ chế **adapted** được cô lập bằng ablation; đóng góp cần bảo vệ vẫn là reliability theo thuộc tính và conflict-aware constraints cho retail.
- Không xem object detection là đóng góp chính của paper GCD.
- Không dùng hyperbolic representation nếu chưa có bằng chứng thực nghiệm.
- Không tuyên bố benchmark mới nếu dữ liệu không thể công bố hoặc protocol chưa đủ chặt chẽ.
- Không sử dụng tên SKU đầy đủ hoặc mã sản phẩm làm input mà không công bố rõ.

---

## 15. Kết luận cuối cùng

Hướng được chốt là:

> **Attribute-level Reliability and Conflict-Aware Generalized Category Discovery for Fine-Grained Retail SKU Recognition.**

Phương pháp khai thác hình ảnh, OCR và các thuộc tính sản phẩm có cấu trúc; ước lượng độ tin cậy riêng cho từng thuộc tính; sau đó sử dụng độ tin cậy và xung đột giữa modality để điều khiển affinity learning và pseudo-labeling.

Giá trị khoa học cốt lõi không nằm ở việc ghép các paper hiện có, mà ở việc chứng minh rằng **reliability ở cấp thuộc tính giúp khám phá đúng các SKU mới trong những nhóm sản phẩm gần giống nhau, ngay cả khi text và hình ảnh bị nhiễu hoặc mâu thuẫn**.

Việc đầu tiên phải làm là pilot experiment `visual-only / text-only / simple fusion / oracle attributes`. Chỉ bắt đầu phát triển kiến trúc đầy đủ sau khi kết quả pilot xác nhận khoảng trống nghiên cứu.
