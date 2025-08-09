import sys
import os
import torch
from PyQt6.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, 
                           QWidget, QFileDialog, QTextEdit, QHBoxLayout, 
                           QComboBox, QFormLayout, QStatusBar, QMainWindow,
                           QGroupBox, QProgressBar, QSplitter, QFrame)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import time

VERSION = "1.0.0"
APP_NAME = "Nagato-Sakura-Image-Classification"
ORGANIZATION_NAME = "Nagato-Sakura-Image-Classification"

def get_version():
    """獲取應用程式版本"""
    return VERSION

def get_app_info():
    """獲取應用程式資訊"""
    return {
        "name": APP_NAME,
        "version": VERSION,
        "organization": ORGANIZATION_NAME
    }

def get_icon_path(version=None):
    """根據版本獲取圖示路徑"""
    if version is None:
        version = VERSION
    icon_filename = f"{version}.ico"
    return os.path.join(os.path.dirname(__file__), 'assets', 'icon', icon_filename)

def setup_application():
    """設置應用程式基本資訊"""
    app = QApplication(sys.argv)
    app_info = get_app_info()
    app.setApplicationName(app_info["name"])
    app.setApplicationVersion(app_info["version"])
    app.setOrganizationName(app_info["organization"])
    icon_path = get_icon_path()
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    return app

def create_welcome_message():
    """創建歡迎訊息"""
    return f"""
歡迎使用長門櫻圖像分類 v{VERSION}！

使用步驟:
1. 選擇模型類型 (efficientnet_b0/mobilenet_v3_small/mobilenet_v2)
2. 載入標籤檔案 (labels.txt)
3. 載入模型檔案 (.pth) 或優化模型 (.pt)
4. 選擇圖片或直接拖拽圖片進行分類

提示: 推薦使用 .pt 格式的優化模型以獲得更快的推理速度！
    """.strip()

class ImageClassifierTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.version = VERSION
        self.initUI()

    def get_window_title(self):
        """獲取視窗標題"""
        return APP_NAME

    def initUI(self):
        icon_path = get_icon_path(self.version)
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle(self.get_window_title())
        self.setGeometry(100, 100, 1000, 700)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('準備就緒 - 請先載入標籤和模型檔案')
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)
        control_panel = self.create_control_panel()
        main_splitter.addWidget(control_panel)
        result_panel = self.create_result_panel()
        main_splitter.addWidget(result_panel)
        main_splitter.setSizes([400, 600])
        self.model = None
        self.class_names = None
        self.img_size = 256
        self.is_model_loaded = False
        self.is_labels_loaded = False
        
    def create_control_panel(self):
        """創建左側控制面板"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        model_group = QGroupBox("模型設置")
        model_layout = QVBoxLayout(model_group)
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel('模型類型:'))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['efficientnet_b0', 'mobilenet_v3_small', 'mobilenet_v2'])
        self.model_type_combo.setToolTip('選擇要使用的模型架構')
        model_type_layout.addWidget(self.model_type_combo)
        model_layout.addLayout(model_type_layout)
        self.load_labels_button = QPushButton('載入標籤檔案')
        self.load_labels_button.clicked.connect(self.load_labels)
        self.load_labels_button.setToolTip('載入包含類別名稱的標籤檔案')
        model_layout.addWidget(self.load_labels_button)
        self.load_model_button = QPushButton('載入模型檔案 (.pth)')
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setToolTip('載入訓練好的PyTorch模型')
        model_layout.addWidget(self.load_model_button)
        self.load_traced_button = QPushButton('載入優化模型 (.pt)')
        self.load_traced_button.clicked.connect(self.load_traced_model)
        self.load_traced_button.setToolTip('載入TorchScript優化模型（推理更快）')
        model_layout.addWidget(self.load_traced_button)
        self.status_labels = QFrame()
        status_layout = QVBoxLayout(self.status_labels)
        status_layout.setContentsMargins(10, 10, 10, 10)
        self.labels_status = QLabel('❌ 標籤檔案: 未載入')
        self.model_status = QLabel('❌ 模型檔案: 未載入')
        status_layout.addWidget(self.labels_status)
        status_layout.addWidget(self.model_status)
        model_layout.addWidget(self.status_labels)
        control_layout.addWidget(model_group)
        classify_group = QGroupBox("圖片分類")
        classify_layout = QVBoxLayout(classify_group)
        self.load_image_button = QPushButton('選擇圖片進行分類')
        self.load_image_button.setEnabled(False)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_image_button.setToolTip('選擇要分類的圖片檔案')
        classify_layout.addWidget(self.load_image_button)
        self.drop_area = QLabel('將圖片拖拽到此處')
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setMinimumHeight(80)
        self.drop_area.setAcceptDrops(True)
        self.drop_area.dragEnterEvent = self.dragEnterEvent
        self.drop_area.dropEvent = self.dropEvent
        classify_layout.addWidget(self.drop_area)
        control_layout.addWidget(classify_group)
        log_group = QGroupBox("操作日誌")
        log_layout = QVBoxLayout(log_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.result_text)
        control_layout.addWidget(log_group)
        control_layout.addStretch()
        return control_widget

    def create_result_panel(self):
        """創建右側結果面板"""
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        image_group = QGroupBox("圖片預覽")
        image_layout = QVBoxLayout(image_group)
        self.image_label = QLabel()
        self.image_label.setFixedSize(320, 320)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText('請選擇要分類的圖片')
        image_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(image_group)
        prediction_group = QGroupBox("分類結果")
        prediction_layout = QVBoxLayout(prediction_group)
        self.class_label = QLabel('預測類別: 待分類')
        self.class_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        prediction_layout.addWidget(self.class_label)
        self.class_accuracy = QLabel('準確度: --')
        self.class_accuracy.setFont(QFont("Arial", 12))
        prediction_layout.addWidget(self.class_accuracy)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prediction_layout.addWidget(self.progress_bar)
        top_label = QLabel("詳細分類結果 (前10名):")
        top_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        prediction_layout.addWidget(top_label)
        self.top_results_text = QTextEdit()
        self.top_results_text.setReadOnly(True)
        self.top_results_text.setFont(QFont("Consolas", 10))
        prediction_layout.addWidget(self.top_results_text)
        result_layout.addWidget(prediction_group)
        return result_widget

    def dragEnterEvent(self, event: QDragEnterEvent):
        """處理拖拽進入事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """處理放置事件"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.result_text.append(f"拖拽載入圖片: {file_path}")
            self.display_image(file_path)
            self.classify_image(file_path)
            self.status_bar.showMessage(f'正在分類圖片: {os.path.basename(file_path)}')

    def update_status_indicators(self):
        """更新狀態指示器"""
        if self.is_labels_loaded:
            self.labels_status.setText('✅ 標籤檔案: 已載入')
        else:
            self.labels_status.setText('❌ 標籤檔案: 未載入')
            
        if self.is_model_loaded:
            self.model_status.setText('✅ 模型檔案: 已載入')
        else:
            self.model_status.setText('❌ 模型檔案: 未載入')
        ready = self.is_model_loaded and self.is_labels_loaded
        self.load_image_button.setEnabled(ready)
        if ready:
            self.drop_area.setText('將圖片拖拽到此處\n✅ 已準備就緒，可以開始分類！')
            self.status_bar.showMessage('✅ 準備就緒 - 可以開始圖片分類')
        else:
            self.drop_area.setText('將圖片拖拽到此處\n⏳ 請先載入模型和標籤檔案')
            missing = []
            if not self.is_labels_loaded:
                missing.append('標籤檔案')
            if not self.is_model_loaded:
                missing.append('模型檔案')
            self.status_bar.showMessage(f'⏳ 請載入: {", ".join(missing)}')

    def show_progress(self, show=True):
        """顯示或隱藏進度條"""
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setRange(0, 0)

    def load_labels(self):
        """載入標籤檔案"""
        labels_path = QFileDialog.getOpenFileName(self, '選擇標籤檔案', 'models', '文本檔案 (*.txt)')[0]
        if labels_path:
            encodings = ['utf-8', 'big5', 'gbk', 'cp950', 'latin-1']
            for encoding in encodings:
                try:
                    with open(labels_path, 'r', encoding=encoding) as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                    self.result_text.append(f"標籤檔案載入成功: {os.path.basename(labels_path)}")
                    self.result_text.append(f"共載入 {len(self.class_names)} 個類別")
                    self.result_text.append(f"使用編碼: {encoding}")
                    self.is_labels_loaded = True
                    self.update_status_indicators()
                    return
                except UnicodeDecodeError:
                    continue
            self.result_text.append("❌ 錯誤: 無法解碼標籤檔案，請檢查檔案編碼")

    def load_model(self):
        """載入PyTorch模型檔案"""
        model_path = QFileDialog.getOpenFileName(self, '選擇模型檔案', 'models', '模型檔案 (*.pth)')[0]
        if model_path:
            try:
                if not self.is_labels_loaded:
                    self.result_text.append("⚠️ 請先載入標籤檔案")
                    return
                self.show_progress(True)
                self.status_bar.showMessage('正在載入模型...')
                model_type = self.model_type_combo.currentText()
                num_classes = len(self.class_names)
                self.result_text.append(f"正在載入 {model_type} 模型...")
                if model_type == 'mobilenet_v3_small':
                    self.model = models.mobilenet_v3_small(weights=None)
                    self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
                elif model_type == 'efficientnet_b0':
                    self.model = models.efficientnet_b0(weights=None)
                    self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
                elif model_type == 'mobilenet_v2':
                    self.model = models.mobilenet_v2(weights=None)
                    self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = self.model.to(device)
                self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                self.model.eval()
                self.result_text.append(f"模型載入成功: {os.path.basename(model_path)}")
                self.result_text.append(f"模型類型: {model_type}")
                self.result_text.append(f"運行裝置: {device.upper()}")
                self.is_model_loaded = True
                self.update_status_indicators()
            except Exception as e:
                self.result_text.append(f"❌ 模型載入失敗: {str(e)}")
            finally:
                self.show_progress(False)

    def load_traced_model(self):
        """載入TorchScript優化模型"""
        model_path = QFileDialog.getOpenFileName(self, '選擇優化模型檔案', 'models', 'TorchScript 模型 (*.pt)')[0]
        if model_path:
            try:
                if not self.is_labels_loaded:
                    self.result_text.append("⚠️ 請先載入標籤檔案")
                    return
                self.show_progress(True)
                self.status_bar.showMessage('正在載入優化模型...')
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = torch.jit.load(model_path, map_location=device)
                self.model.eval()
                self.result_text.append(f"優化模型載入成功: {os.path.basename(model_path)}")
                self.result_text.append(f"TorchScript優化模型 - 推理更快")
                self.result_text.append(f"運行裝置: {device.upper()}")
                self.is_model_loaded = True
                self.update_status_indicators()
            except Exception as e:
                self.result_text.append(f"❌ 優化模型載入失敗: {str(e)}")
            finally:
                self.show_progress(False)

    def load_image(self):
        """載入並分類圖片"""
        image_path = QFileDialog.getOpenFileName(
            self, '選擇圖片檔案', '', 
            '圖片檔案 (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;所有檔案 (*.*)')[0]
        if image_path:
            self.result_text.append(f"已選擇圖片: {os.path.basename(image_path)}")
            self.display_image(image_path)
            self.classify_image(image_path)

    def display_image(self, image_path):
        """顯示圖片在界面上"""
        try:
            image = Image.open(image_path).convert('RGB')
            display_size = 320
            image.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
            background = Image.new('RGB', (display_size, display_size), (255, 255, 255))
            offset = ((display_size - image.width) // 2, (display_size - image.height) // 2)
            background.paste(image, offset)
            qimage = QImage(background.tobytes(), 
                          background.width, background.height, 
                          3 * background.width, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.result_text.append(f"❌ 顯示圖片失敗: {str(e)}")
            self.image_label.setText('圖片顯示失敗')

    def classify_image(self, image_path):
        """對圖片進行分類"""
        if self.model is None or self.class_names is None:
            self.result_text.append("⚠️ 請先載入模型和標籤檔案")
            return
        try:
            start_time = time.time()
            self.show_progress(True)
            self.status_bar.showMessage('正在進行圖片分類...')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probs, indices = torch.sort(probabilities, descending=True)
                probs = probs[0].cpu().numpy()
                indices = indices[0].cpu().numpy()
            inference_time = time.time() - start_time
            top_idx = indices[0]
            top_prob = probs[0] * 100
            top_class = self.class_names[top_idx]
            self.class_label.setText(f"預測類別: {top_class}")
            if top_prob >= 80:
                confidence_emoji = "🟢"
            elif top_prob >= 60:
                confidence_emoji = "🟡"
            else:
                confidence_emoji = "🔴"
            self.class_accuracy.setText(f"{confidence_emoji} 準確度: {top_prob:.2f}%")
            self.top_results_text.clear()
            self.top_results_text.append("分類結果排行榜:\n" + "="*40)
            for i in range(min(len(self.class_names), 10)):
                idx = indices[i]
                prob = probs[i] * 100
                class_name = self.class_names[idx]
                rank = f"{i+1:2}."
                bar_length = 20
                filled = int((prob / 100) * bar_length)
                bar = "█" * filled + " " * (bar_length - filled)
                self.top_results_text.append(
                    f"{rank} {class_name:<20} {prob:6.2f}% {bar}")
            self.top_results_text.append("\n" + "="*40)
            self.top_results_text.append(f"推理時間: {inference_time:.3f} 秒")
            self.top_results_text.append(f"運算裝置: {device.upper()}")
            self.top_results_text.append(f"圖片尺寸: {image.size[0]}x{image.size[1]}")
            self.result_text.append(f"分類完成: {top_class} ({top_prob:.2f}%)")
            self.result_text.append(f"推理時間: {inference_time:.3f}秒")
            self.status_bar.showMessage(f'✅ 分類完成: {top_class} ({top_prob:.2f}%)')
        except Exception as e:
            self.result_text.append(f"❌ 分類失敗: {str(e)}")
            import traceback
            self.result_text.append(f"詳細錯誤: {traceback.format_exc()}")
            self.status_bar.showMessage('❌ 分類失敗')
        finally:
            self.show_progress(False)

def main():
    """主函數"""
    app = setup_application()
    window = ImageClassifierTester()
    window.show()
    welcome_msg = create_welcome_message()
    window.result_text.append(welcome_msg)
    sys.exit(app.exec())
if __name__ == '__main__':
    main()