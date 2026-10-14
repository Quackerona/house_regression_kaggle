# Phân tích dữ liệu khám phá (EDA)

**Đề tài:** Kaggle: House Prices - Advanced Regression Techniques
**Dữ liệu:** thông tin các căn nhà ở Ames, Iowa (`train.csv` / `test.csv`)
**Biến mục tiêu:** `SalePrice` (giá bán của căn nhà, đô la Mỹ)

## Tóm tắt tổng quan

Báo cáo này đi qua dữ liệu giá nhà từ lúc rửa dữ liệu tới lúc chọn feature cho model, dựa trên 1460 căn nhà trong tập train. Các phát hiện dưới đây được dùng làm cơ sở để xử lý dữ liệu và huấn luyện ở bước tiếp theo.

### Các phát hiện chính:

- Cột số nào cũng lệch phải, kể cả `SalePrice`, nên mình log hết bằng `log1p`.
- Nhiều cột diện tích gần như toàn số 0 (hồ bơi, hiên, tầng hầm phụ...), mấy cột này không có ích gì nhiều.
- Trong 79 cột gốc, có 25 cột mà SHAP gần như bằng 0 khi train baseline. Bỏ đi 35 cột gốc và thêm vài cột mới thì RMSE giảm rõ.
- `TotalSF` là cột mình tự tạo, tương quan với giá tới 0.81 và thành feature mạnh nhất sau khi train lại.
- `OverallQual` (chất lượng tổng thể) và `Neighborhood` (khu vực) luôn nằm trong top 3, cả trước lẫn sau khi tạo feature.
- RMSE trên validation từ 0.13560 xuống 0.12482 sau feature engineering. Nộp lên Kaggle được 0.12622.

## 1. Sơ lược về dữ liệu

Tập train gồm **1460 dòng** và **81 cột**: 1 biến mục tiêu (`SalePrice`), 79 biến đặc trưng, và 1 mã định danh (`Id`).

79 cột đặc trưng khá nhiều nên mình chia theo nhóm cho dễ nhìn:

| Nhóm | Cột ví dụ | Ghi chú |
| :--- | :--- | :--- |
| Lô đất và vị trí | `MSZoning`, `LotFrontage`, `LotArea`, `Neighborhood`, `LotShape` | `Neighborhood` là tên khu phố |
| Kiểu nhà | `MSSubClass`, `BldgType`, `HouseStyle`, `RoofStyle` | Toàn là chữ hoặc mã loại |
| Chất lượng và tình trạng | `OverallQual`, `OverallCond`, `ExterQual`, `KitchenQual`, `HeatingQC` | Điểm từ 1 đến 10 hoặc các mức Ex, Gd, TA, Fa, Po |
| Năm | `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `YrSold`, `MoSold` | Năm xây, năm sửa, năm bán |
| Diện tích | `GrLivArea`, `1stFlrSF`, `2ndFlrSF`, `TotalBsmtSF`, `GarageArea` | Đơn vị là feet vuông |
| Số phòng | `FullBath`, `HalfBath`, `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd` | Cột đếm |
| Tầng hầm | `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinSF1` | Nhà không có hầm thì ghi `NA` |
| Gara | `GarageType`, `GarageFinish`, `GarageCars`, `GarageQual` | Nhà không có gara thì ghi `NA` |
| Hiên, hồ bơi, linh tinh | `WoodDeckSF`, `OpenPorchSF`, `PoolArea`, `PoolQC`, `Fence`, `MiscFeature` | Hầu hết nhà không có |
| Giao dịch | `SaleType`, `SaleCondition` | Cách thức bán |

Có một cái bẫy ở bộ dữ liệu này: rất nhiều cột chữ có giá trị `NA`, nhưng đó không phải dữ liệu thiếu. Nó có nghĩa là nhà không có thứ đó (không có hồ bơi, không có gara, không có ngõ hẻm...). Pandas mặc định đọc `NA` thành NaN nên mình tắt mặc định đó đi, chỉ cho `NA` thành NaN ở những cột mà chuyện "không có" không thể xảy ra (`mandatory_cols`).

## 2. Xử lý dữ liệu thiếu

train.csv:

| Cột | Số thiếu | Cách giải quyết |
| :--- | :--- | :--- |
| `LotFrontage` | 259 | Điền bằng trung vị. |
| `MasVnrArea` | 8 | Điền bằng trung vị. |
| `GarageYrBlt` | 81 | Điền bằng `YearBuilt`, vì `GarageType` vẫn cho biết nhà đó không có gara. |
| `Electrical` | 1 | Điền `SBrkr`, giá trị này chiếm 91% nên chọn luôn. |

Mấy chỗ khác cần sửa thêm trong train:

- `MasVnrType` có 2 kiểu để chỉ "không có" là `NA` và `None`. Mình đổi hết `None` thành `NA`.
- `LotFrontage`, `MasVnrArea`, `GarageYrBlt` bị đọc thành chữ vì có ô `NA` lẫn vào, nên phải ép lại thành số.

test.csv:

| Cột | Cách giải quyết |
| :--- | :--- |
| `LotFrontage`, `MasVnrArea` | Điền bằng trung vị của train, không tính lại trên test. |
| `GarageYrBlt` | Làm giống train, điền bằng `YearBuilt`. |
| `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`, `GarageCars`, `GarageArea` | Mấy cột này train không bị thiếu nhưng test thì có vài dòng thiếu. Ô trống ở đây nghĩa là không có hầm hoặc gara nên điền 0. |
| `KitchenQual`, `Exterior1st`, `Exterior2nd` | Điền bằng giá trị xuất hiện nhiều nhất trong train. |

Test còn thiếu vài ô ở `MSZoning`, `Utilities`, `Functional`, `SaleType`, nhưng 4 cột này bị bỏ ở bước chọn feature nên không cần điền.

## 3. Kiểm tra skewness và outliers

Mình vẽ histogram cho 20 cột số liên tục (diện tích và giá trị), bỏ các cột chỉ để đếm (số phòng, số xe, số lò sưởi) và các cột là mã hoặc năm. Những gì thấy được:

- Hầu hết đều lệch phải, đuôi dài về phía giá trị lớn.
- `SalePrice` kéo dài tới hơn 700000 trong khi phần lớn nhà nằm quanh 100000 tới 250000.
- `LotArea` có vài lô đất rất lớn, hơn 200000, trong khi phần lớn dưới 20000.
- `PoolArea`, `MiscVal`, `3SsnPorch`, `LowQualFinSF`, `ScreenPorch`, `BsmtFinSF2`, `EnclosedPorch` gần như toàn số 0, chỉ có vài căn có giá trị.

Mình không bỏ dòng nào. Cách xử lý là chuyển hết 20 cột (cả `SalePrice`) sang log. Vì có nhiều giá trị 0 nên dùng `log1p` thay vì log thường để không bị lỗi.

Vì `SalePrice` đã bị log, model dự đoán ra giá ở dạng log, phải dùng `expm1` để đổi lại thành giá thật. RMSE ở dưới cũng tính trên thang log, giống cách Kaggle chấm điểm.

## 4. Tạo feature mới

Trước khi tính, mình đưa các cột diện tích về đơn vị gốc bằng `expm1`, vì cộng hay chia số đã log sẽ ra kết quả sai. Sau đó tạo 11 cột mới, là những thứ người ta hay để ý khi đánh giá một căn nhà.

| Feature | Cách tạo | Tương quan với `SalePrice` |
| :--- | :--- | :--- |
| `TotalSF` | Hầm + tầng 1 + tầng 2, rồi log | 0.807 |
| `TotalBath` | Phòng tắm đầy đủ + 0.5 lần phòng tắm nhỏ, cả trên mặt đất lẫn dưới hầm | 0.673 |
| `AreaPerRoom` | `GrLivArea` chia `TotRmsAbvGrd`, rồi log | 0.576 |
| `HasWoodDeck` | Có sàn gỗ ngoài trời hay không | 0.320 |
| `TotalPorchSF` | Tổng 4 loại hiên, rồi log | 0.310 |
| `HasScreenPorch` | Có hiên lưới hay không | 0.099 |
| `LotCoverage` | Tầng 1 chia diện tích lô đất | -0.023 |
| `IsRemodeled` | Năm sửa khác năm xây hay không | -0.074 |
| `HasEnclosedPorch` | Có hiên kín hay không | -0.217 |
| `RemodAge` | Năm bán trừ năm sửa | -0.568 |
| `HouseAge` | Năm bán trừ năm xây | -0.587 |

Vài điều rút ra:

- `TotalSF` mạnh nhất với 0.807.
- `HouseAge` và `RemodAge` đều âm khoảng 0.57 đến 0.59, nhà càng cũ giá càng thấp.
- `LotCoverage` và `IsRemodeled` gần như bằng 0 nên bỏ luôn.
- Con số này chỉ đo quan hệ đường thẳng với giá đã log, dùng để tham khảo thôi. Cột nào giữ hay bỏ vẫn phải xem SHAP sau khi train.

## 5. Chọn feature bằng SHAP

Mình dùng FLAML AutoML để train, rồi xem SHAP để biết cột nào model thực sự dùng.

### Baseline (79 feature gốc)

Thuật toán được chọn là `lgbm`, RMSE trên validation là **0.13560**. Top đầu theo SHAP:

| Hạng | Cột | SHAP |
| :--- | :--- | :--- |
| 1 | `OverallQual` | 0.10 |
| 2 | `GrLivArea` | 0.08 |
| 3 | `Neighborhood` | 0.06 |
| 4 | `TotalBsmtSF` | 0.03 |
| 5 | `OverallCond` | 0.03 |

25 cột cuối bảng có SHAP gần như bằng 0: `ExterQual`, `LotShape`, `MSZoning`, `LandSlope`, `Condition2`, `Alley`, `Street`, `LandContour`, `Utilities`, `RoofMatl`, `BldgType`, `GarageCond`, `GarageQual`, `Functional`, `Heating`, `BsmtCond`, `PoolQC`, `MiscFeature`, `BsmtHalfBath`, `BsmtFinSF2`, `LowQualFinSF`, `KitchenAbvGr`, `3SsnPorch`, `MiscVal`, `PoolArea`.

Trong 25 cột này mình bỏ 24, riêng `ExterQual` được giữ lại. Ngoài ra bỏ thêm 11 cột có SHAP rất nhỏ ở phần giữa bảng: `Condition1`, `ExterCond`, `MasVnrType`, `Electrical`, `BsmtFinType2`, `RoofStyle`, `PavedDrive`, `Fence`, `YrSold`, `MoSold`, `SaleType`. Cộng với 2 feature mới yếu (`LotCoverage`, `IsRemodeled`), tổng cộng bỏ 37 cột, còn lại **53 feature**.

### Sau khi tạo feature và bỏ cột (53 feature)

Vẫn là `lgbm`, RMSE trên validation giảm xuống **0.12482**. Top đầu theo SHAP:

| Hạng | Cột | SHAP |
| :--- | :--- | :--- |
| 1 | `TotalSF` | 0.10 |
| 2 | `OverallQual` | 0.08 |
| 3 | `Neighborhood` | 0.06 |
| 4 | `TotalBath` | 0.03 |

Có mấy điều đáng để ý:

- **`TotalSF` lên hạng 1**, đẩy `OverallQual` xuống hạng 2. Đúng với dự đoán ở mục 4 là cột này mạnh.
- **`GrLivArea` tụt từ 0.08 xuống 0.02, `TotalBsmtSF` tụt từ 0.03 xuống gần 0.** Khả năng cao là vì `TotalSF` đã gom gần hết diện tích của mấy cột đó, model chỉ cần dùng một cột là đủ. Tương tự, `TotalBath` (0.03) làm `FullBath` và `BsmtFullBath` gần như hết việc.
- **`HouseAge` và `RemodAge` đều có SHAP khoảng 0.01**, nằm trong nhóm đầu bảng. Tuổi nhà thực sự có ích.
- **`HasWoodDeck`, `HasScreenPorch`, `HasEnclosedPorch` gần như bằng 0** dù có tương quan nhìn cũng ổn ở mục 4. Mình vẫn để lại vì chúng không ảnh hưởng gì xấu, nhưng nếu cần cắt thêm thì bắt đầu từ mấy cột này.

### Kết quả

| Bước | RMSE |
| :--- | :--- |
| Baseline, 79 feature, validation (20% train) | 0.13560 |
| Sau feature engineering, 53 feature, validation (20% train) | 0.12482 |
| Nộp Kaggle, train trên toàn bộ train | 0.12622 |

Chênh lệch giữa validation và Kaggle chỉ khoảng 0.0014, nhỏ, không có dấu hiệu overfit. RMSE 0.126 trên thang log tương đương sai khoảng 12 đến 13% giá nhà.

## 6. Đề xuất sửa dữ liệu

Để dữ liệu sạch và dễ cho model, mình làm theo thứ tự này:

- **Bỏ cột `Id`** vì không mang thông tin gì.
- **Đổi `None` thành `NA` ở `MasVnrType`** cho đồng nhất, và ép 3 cột `LotFrontage`, `MasVnrArea`, `GarageYrBlt` về dạng số.
- **Điền dữ liệu thiếu** theo mục 2, trung vị và giá trị phổ biến nhất đều lấy từ train.
- **Log hết 20 cột số liên tục** bằng `log1p`, kể cả `SalePrice`.
- **Tạo 11 feature mới** như ở mục 4, nhớ `expm1` các cột diện tích trước khi tính.
- **Bỏ 37 cột** theo SHAP và tương quan ở mục 5.
- **Test làm đúng y như train**, cùng thứ tự cột, cùng danh sách cột bị bỏ.

## 7. Kết luận rút ra từ EDA

- **Dữ liệu này lệch rất nhiều và nhiều cột gần như vô dụng.** Log giải quyết phần lệch, còn SHAP giúp tìm ra 35 cột gốc có thể bỏ mà không mất gì.
- **Tự tạo feature có hiệu quả thật.** `TotalSF` và `TotalBath` đều vào top 4, và RMSE giảm từ 0.13560 xuống 0.12482 sau bước này.
- **Feature mới cũng làm vài feature cũ hết giá trị.** `GrLivArea` và `TotalBsmtSF` yếu hẳn đi sau khi có `TotalSF`. Tương quan cao chưa chắc model sẽ dùng, phải xem SHAP mới biết chắc.
- **`OverallQual` và `Neighborhood` luôn quan trọng.** Hai cột này đứng top 3 ở cả hai lần train, dù có thêm hay bớt cột nào.
- **Kết quả ổn định.** Validation 0.12482 và Kaggle 0.12622 gần nhau, nên pipeline này đáng tin.

## 8. Triển khai model

- Dùng **FLAML AutoML** với `time_budget=120` giây, metric `rmse`, `seed=42`. Cả 2 lần train đều chọn ra `lgbm`.
- Không encode cột chữ bằng tay, để FLAML tự xử lý.
- Ở bước thử, chia 80% train và 20% validation với `random_state=42` để đo RMSE.
- Ở bước cuối (notebook 05), train trên toàn bộ train, xử lý test giống hệt train, rồi dự đoán và dùng `expm1` để đổi về giá thật.
- Mỗi lần chạy xuất ra một file mới trong `data/results` theo ngày giờ, không ghi đè.
- Kết quả: RMSE 0.12482 (validation), 0.12622 (Kaggle).
