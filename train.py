import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='高效能圖片分類模型訓練')
    parser.add_argument('--dataset', default='data', help='訓練資料集資料夾路徑')
    parser.add_argument('--save_dir', default='models', help='模型儲存目錄')
    parser.add_argument('--epochs', type=int, default=1000, help='訓練回合數')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='學習率')
    parser.add_argument('--model', type=str, default='efficientnet_b0', 
                      choices=['mobilenet_v3_small', 'efficientnet_b0', 'mobilenet_v2'], 
                      help='模型架構')
    parser.add_argument('--img_size', type=int, default=256, help='圖片尺寸')
    parser.add_argument('--val_split', type=float, default=0.1, help='驗證集比例')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用裝置: {device}")
    os.makedirs(args.save_dir, exist_ok=True)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print(f"正在載入資料集: {args.dataset}")
    full_dataset = datasets.ImageFolder(root=args.dataset, transform=train_transform)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"發現 {num_classes} 個類別: {class_names}")
    with open(os.path.join(args.save_dir, "labels.txt"), 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"類別名稱已儲存至 {os.path.join(args.save_dir, 'labels.txt')}")
    print(f"創建 {args.model} 模型...")
    if args.model == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif args.model == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif args.model == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print("開始訓練...")
    best_acc = 0.0
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    start_time = time.time()
    for epoch in range(args.epochs):
        print(f'回合 {epoch+1}/{args.epochs}')
        print('-' * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        pbar = tqdm(train_loader, desc=f'訓練中')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pbar.set_postfix({'loss': loss.item()})
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f'訓練損失: {epoch_loss:.4f} 準確度: {epoch_acc:.4f}')
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'驗證中')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)
        print(f'驗證損失: {val_loss:.4f} 準確度: {val_acc:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'最佳模型已保存，準確度: {best_acc:.4f}')
        scheduler.step()
    total_time = time.time() - start_time
    print(f"訓練完成，耗時: {total_time // 60:.0f}分 {total_time % 60:.0f}秒")
    print("最終處理和儲存模型...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    example_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model_path = os.path.join(args.save_dir, "model_traced.pt")
    traced_model.save(traced_model_path)
    print(f"最佳化追蹤模型已儲存至 {traced_model_path}")
    model_path = os.path.join(args.save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"原始模型已儲存至 {model_path}")
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    traced_size = os.path.getsize(traced_model_path) / (1024 * 1024)
    print(f"模型大小比較:")
    print(f"  原始模型: {orig_size:.2f} MB")
    print(f"  最佳化模型: {traced_size:.2f} MB")
    print(f"  節省空間: {(1 - traced_size/orig_size) * 100:.1f}%")
if __name__ == "__main__":
    main()