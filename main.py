import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QProgressBar, QTextEdit, QSpinBox, QDoubleSpinBox, QFormLayout, QHBoxLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon

# 設置運算裝置、檢測環境
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"<環境檢測>\nPyTorch 版本: {torch.__version__}\n運行模式: {device}")
current_directory = Path(__file__).resolve().parent

# 模型訓練的工作線程
class TrainThread(QThread):
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    
    def __init__(self, train_loader, model, criterion, optimizer, num_epochs, batch_size, learning_rate, save_path, class_names):
        super().__init__()
        self.train_loader = train_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.class_names = class_names
        self.total_steps = len(self.train_loader) * num_epochs

    def run(self):
        current_step = 0
        start_time = time.time()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(self.train_loader):
                batch_start_time = time.time()
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()  # 損失累積
                current_step += 1
                progress_percent = int((current_step / self.total_steps) * 100)  # 計算進度百分比
                self.progress.emit(progress_percent)
                batch_end_time = time.time()  # 計算時間
                batch_time = batch_end_time - batch_start_time
                elapsed_time = batch_end_time - start_time
                avg_time_per_step = elapsed_time / current_step
                estimated_total_time = avg_time_per_step * self.total_steps
                remaining_time = estimated_total_time - elapsed_time
                
            self.log_signal.emit(
                f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss / (i+1):.4f}, [{self.format_time(elapsed_time)}>{self.format_time(remaining_time)}, {batch_time:.2f}s/it]'
            )

        self.log_signal.emit("訓練完成！")
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pth"))
        self.log_signal.emit(f"模型已保存到 {os.path.join(self.save_path, 'model.pth')}")
        with open(os.path.join(self.save_path, "labels.txt"), 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        self.log_signal.emit(f"類別名稱已保存到 {os.path.join(self.save_path, 'labels.txt')}")

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

# PyQt6 GUI 應用程式
class ImageClassifierApp(QWidget):
    version = "0.0.1"
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(f"Nagato-Sakura-Image-Classification")
        self.setGeometry(100, 100, 500, 600)
        icon_path = current_directory / "assets" / "icon" / f"{self.version}.ico"
        self.setWindowIcon(QIcon(str(icon_path)))
        
        # 加載數據按鈕
        self.load_button = QPushButton('加載數據')
        self.load_button.clicked.connect(self.load_dataset)
        # 開始訓練按鈕
        self.train_button = QPushButton('開始訓練')
        self.train_button.setEnabled(False)  # 初始時禁用，等加載數據後啟用
        self.train_button.clicked.connect(self.start_training)
        # 訓練參數控制
        self.epochs_spinbox = QSpinBox(self)
        self.epochs_spinbox.setRange(1, 150)
        self.epochs_spinbox.setValue(50)
        self.batch_size_spinbox = QSpinBox(self)
        self.batch_size_spinbox.setRange(1, 128)
        self.batch_size_spinbox.setValue(16)
        self.learning_rate_spinbox = QDoubleSpinBox(self)
        self.learning_rate_spinbox.setRange(0.0001, 1.0)
        self.learning_rate_spinbox.setValue(0.001)
        self.learning_rate_spinbox.setDecimals(6)
        # 顯示訓練進度的進度條
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        # 訓練日誌窗口
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        # 載入類別按鈕
        self.load_labels_button = QPushButton('載入類別')
        self.load_labels_button.clicked.connect(self.load_labels)
        # 加載模型按鈕
        self.load_model_button = QPushButton('載入模型')
        self.load_model_button.setEnabled(False)  # 先禁用，等類別載入後啟用
        self.load_model_button.clicked.connect(self.load_model)
        # 選擇圖片按鈕
        self.load_image_button = QPushButton('選擇圖片')
        self.load_image_button.setEnabled(False)  # 初始禁用
        self.load_image_button.clicked.connect(self.load_image)
        # 顯示分類結果
        self.class_label = QLabel('預測分類:', self)
        self.class_accuracy = QLabel('準確度:', self)
        # 顯示結果的文本框
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        # 顯示選擇的圖片
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(224, 224)
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        params_layout = QFormLayout()
        params_layout.addRow('Epochs:', self.epochs_spinbox)
        params_layout.addRow('Batch Size:', self.batch_size_spinbox)
        params_layout.addRow('Learning Rate:', self.learning_rate_spinbox)
        layout.addLayout(params_layout)
        layout.addWidget(self.train_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_text)
        layout.addWidget(self.load_labels_button)
        layout.addWidget(self.load_model_button)
        layout.addWidget(self.load_image_button)
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.class_label)
        result_layout.addWidget(self.class_accuracy)
        layout.addLayout(result_layout)
        image_result_layout = QHBoxLayout()
        image_result_layout.addWidget(self.image_label)
        image_result_layout.addWidget(self.result_text)
        layout.addLayout(image_result_layout)
        
        self.setLayout(layout)
        self.dataset = None
        self.model = None
        self.class_names = None
        
    def load_dataset(self):
        dataset_dir = QFileDialog.getExistingDirectory(self, '選擇數據夾')
        if dataset_dir:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            train_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size_spinbox.value(), shuffle=True)
            self.class_names = train_dataset.classes
            self.log_text.append(f"數據集從 {dataset_dir} 加載完成")
            self.train_button.setEnabled(True)

    def start_training(self):
        save_dir = QFileDialog.getExistingDirectory(self, '選擇保存文件夾')
        if not save_dir:
            self.log_text.append("請選擇模型保存的文件夾！")
            return
        
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_spinbox.value())
        self.train_thread = TrainThread(self.train_loader, self.model, criterion, optimizer, num_epochs=self.epochs_spinbox.value(),batch_size=self.batch_size_spinbox.value(),learning_rate=self.learning_rate_spinbox.value(),save_path=save_dir, class_names=self.class_names)
        self.train_thread.progress.connect(self.update_progress)
        self.train_thread.log_signal.connect(self.update_log)
        self.train_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_log(self, log_message):
        self.log_text.append(log_message)

    def load_labels(self):
        labels_path = QFileDialog.getOpenFileName(self, '選擇類別名稱文件', '', '文本文件 (*.txt)')[0]
        if labels_path:
            with open(labels_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            self.result_text.append(f"類別名稱從 {labels_path} 加載完成")
            self.load_model_button.setEnabled(True)

    def load_model(self):
        model_path = QFileDialog.getOpenFileName(self, '選擇模型文件', '', '模型文件 (*.pth)')[0]
        if model_path:
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.result_text.append(f"模型從 {model_path} 加載完成")
            self.load_image_button.setEnabled(True)

    def load_image(self):
        image_path = QFileDialog.getOpenFileName(self, '選擇圖片文件', '', '圖像 (*.png *.jpg *.jpeg)')[0]
        if image_path:
            self.result_text.append(f"選擇的圖片: {image_path}")
            self.display_image(image_path)
            self.classify_image(image_path)

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image_qt = QImage(image.tobytes(), image.size[0], image.size[1], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image_qt)
        self.image_label.setPixmap(pixmap)

    def classify_image(self, image_path):
        if self.model is None or self.class_names is None:
            self.result_text.append("請先加載模型和類別名稱。")
            return

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            accuracy = probabilities[0][predicted].item() * 100

        class_name = self.class_names[predicted.item()]
        self.class_label.setText(f"預測分類: {class_name}")
        self.class_accuracy.setText(f"準確度: {accuracy:.2f}%")

# 主函數
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())