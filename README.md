# 🤚 Hand Sign Language Recognition System

Hệ thống nhận dạng ngôn ngữ ký hiệu bằng tay sử dụng Computer Vision và Machine Learning.

## 📋 Mô tả dự án

Dự án này xây dựng một hệ thống tự động nhận dạng và phân loại các cử chỉ ngôn ngữ ký hiệu từ video. Hệ thống sử dụng:
- **MediaPipe Hands** để phát hiện và trích xuất các điểm mốc (landmarks) của bàn tay
- **Machine Learning models** (Random Forest và Neural Networks) để phân loại các cử chỉ
- **Computer Vision** để xử lý và phân tích video đầu vào

## 🎯 Tính năng chính

- ✅ Phát hiện và tracking bàn tay trong video real-time
- ✅ Trích xuất 21 điểm mốc (landmarks) cho mỗi bàn tay (tối đa 2 tay)
- ✅ Phân loại các cử chỉ ngôn ngữ ký hiệu
- ✅ Hỗ trợ cả ngôn ngữ ký hiệu chữ cái và từ ngữ
- ✅ Visualize skeleton của bàn tay trên nền trắng
- ✅ Augmentation dữ liệu (noise, rotation, scaling) để cải thiện độ chính xác

## 🛠️ Công nghệ sử dụng

### Libraries & Frameworks
- **OpenCV**: Xử lý video và hình ảnh
- **MediaPipe**: Phát hiện và tracking bàn tay
- **PyTorch**: Xây dựng mô hình Neural Network
- **Scikit-learn**: Random Forest và các công cụ ML
- **Pandas & NumPy**: Xử lý dữ liệu
- **Matplotlib**: Visualize kết quả

### Models
1. **Random Forest Classifier** - Phân loại dựa trên hand landmarks
2. **Neural Network (LSTM/CNN)** - Deep learning cho sequence classification

## 📁 Cấu trúc dự án

```
xlnnkh/
│
├── xlnn1.ipynb                 # Neural Network implementation
├── random forest.ipynb         # Random Forest classifier
│
├── hand.csv                    # Hand landmarks data (126 features)
├── public_label_clean1.csv    # Labels cho Vietnamese sign language
│
├── dataset/                    # Dataset được chia train/val/test
│   ├── train1.csv
│   ├── validation1.csv
│   ├── test1.csv
│   ├── train/
│   ├── val/
│   └── test/
│
├── public_train/              # Dữ liệu training gốc
│   └── public_train_label.csv
│
└── public_test/               # Dữ liệu testing
```

## 🔢 Dữ liệu

### Hand Landmarks Features
Mỗi sample chứa **126 features**:
- **63 features** cho tay trái (Left hand): 21 landmarks × 3 tọa độ (x, y, z)
- **63 features** cho tay phải (Right hand): 21 landmarks × 3 tọa độ (x, y, z)

### Labels
- **Chữ cái**: Các ký tự trong bảng chữ cái
- **Từ ngữ**: Các từ thông dụng như "an", "ban khoan", "chan", "biet", "cham", v.v.
- **Số**: 0-9

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
pip install opencv-python mediapipe torch torchvision
pip install scikit-learn pandas numpy matplotlib tqdm
```

### 2. Chuẩn bị dữ liệu

```python
# Chia dataset thành train/val/test
split_videos_by_label(
    csv_path="path/to/labels.csv",
    video_dir="path/to/videos",
    output_dir="path/to/output",
    seed=42
)
```

### 3. Trích xuất Hand Landmarks

```python
# Trích xuất keypoints từ video
extract_and_save_keypoints(
    video_path="path/to/video.mp4",
    output_csv="hand.csv",
    max_num_hands=2
)
```

### 4. Training Model

#### Random Forest
```python
# Mở random forest.ipynb và chạy các cells
# Model sẽ được lưu vào file .pkl
```

#### Neural Network
```python
# Mở xlnn1.ipynb và chạy các cells
# Model sẽ được train với PyTorch
```

### 5. Inference

```python
# Load model và predict
model = load_model("model.pkl")  # hoặc .pth cho NN
predictions = model.predict(hand_landmarks)
```

## 📊 Data Augmentation

Dự án implement các kỹ thuật augmentation:
- **Gaussian Noise**: Thêm nhiễu ngẫu nhiên vào tọa độ
- **Rotation**: Xoay bàn tay
- **Scaling**: Thay đổi kích thước
- **Translation**: Dịch chuyển vị trí

## 🎥 Demo

Hệ thống hiển thị 2 khung hình song song:
- **Bên trái**: Video gốc với hand landmarks
- **Bên phải**: Skeleton trên nền trắng

Nhấn `q` để thoát khỏi visualization.

## 📈 Kết quả

Model được đánh giá dựa trên:
- **Accuracy**: Độ chính xác tổng thể
- **Classification Report**: Precision, Recall, F1-score cho từng class
- **Confusion Matrix**: Ma trận nhầm lẫn
- **Log Loss**: Cross-entropy loss

## 🔧 Preprocessing Pipeline

1. **Video Input** → Đọc từng frame
2. **Hand Detection** → MediaPipe phát hiện bàn tay
3. **Landmark Extraction** → Trích xuất 21 điểm × 2 tay
4. **Normalization** → Chuẩn hóa tọa độ
5. **Feature Engineering** → Tạo features bổ sung
6. **Classification** → Dự đoán class

## 🤝 Đóng góp

Nếu bạn muốn đóng góp vào dự án:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 Notes

- Dự án hỗ trợ tối đa **2 bàn tay** đồng thời
- Độ tin cậy tối thiểu cho detection: **0.3-0.5**
- Video được resize về **640×480** trước khi xử lý
- Dataset được chia theo tỷ lệ: **60% train / 20% val / 20% test**

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **CSV có filename trùng**: Tự động loại bỏ duplicates
2. **Class có ít hơn 2 samples**: Bỏ qua class đó khi split
3. **Video không tìm thấy**: Kiểm tra đường dẫn
4. **Không detect được tay**: Tăng `min_detection_confidence`

## 📄 License

Dự án này thuộc về công ty HBD.

## 👥 Tác giả

Được phát triển bởi nhóm AI - Công ty HBD

## 📧 Liên hệ

Nếu có thắc mắc hoặc đề xuất, vui lòng liên hệ qua email hoặc tạo issue trên GitHub.

---

**Happy Coding! 🚀**
