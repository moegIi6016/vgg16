import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


def save_split_images(dataset, split_dataset, split_name):
    """
    データセットの分割結果を保存する関数です。

    Args:
        dataset (torchvision.datasets.ImageFolder): 元のデータセット
        split_dataset (torch.utils.data.Subset): 分割後のデータセット
        split_name (str): 分割の名前

    Returns:
        None
    """
    split_dir = os.path.join(os.path.dirname(dataset.root), split_name)
    os.makedirs(split_dir, exist_ok=True)

    for idx in split_dataset.indices:
        img_path = dataset.imgs[idx][0]
        class_name = dataset.classes[dataset.imgs[idx][1]]
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img_file_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(class_dir, img_file_name))


def load_or_split_dataset(root_dir, data_transforms):
    """
    データセットを読み込むか分割する関数です。

    Args:
        root_dir (str): データセットのルートディレクトリ
        data_transforms (dict): データの前処理を行うための変換関数の辞書

    Returns:
        tuple: 訓練データセット、検証データセット、テストデータセットのタプル
    """
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['val'])
    return train_dataset, val_dataset, test_dataset


def calculate_class_accuracies(predicted_labels, true_labels, class_names):
    """
    各クラスの正解率を計算する関数です。

    Args:
        predicted_labels (list): 予測されたラベルのリスト
        true_labels (list): 正解ラベルのリスト
        class_names (list): クラス名のリスト

    Returns:
        dict: クラス名をキー、正解率を値とする辞書
    """
    class_counts = defaultdict(lambda: [0, 0])

    for pred, true in zip(predicted_labels, true_labels):
        class_name = class_names[true]
        class_counts[class_name][1] += 1
        if pred == true:
            class_counts[class_name][0] += 1

    class_accuracies = {}
    for class_name, counts in class_counts.items():
        class_accuracies[class_name] = counts[0] / counts[1]

    return class_accuracies


def train(model, dataloader, criterion, optimizer, device):
    """
    モデルの訓練を行う関数です。

    Args:
        model (torch.nn.Module): 訓練するモデル
        dataloader (torch.utils.data.DataLoader): 訓練データを提供するデータローダー
        criterion (torch.nn.Module): 損失関数
        optimizer (torch.optim.Optimizer): 最適化手法
        device (torch.device): デバイス (CPUまたはGPU)

    Returns:
        float, float: エポックの損失値と正解率
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    モデルの評価を行う関数です。

    Args:
        model (torch.nn.Module): 評価するモデル
        dataloader (torch.utils.data.DataLoader): 評価データを提供するデータローダー
        criterion (torch.nn.Module): 損失関数
        device (torch.device): デバイス (CPUまたはGPU)

    Returns:
        float, float: エポックの損失値と正解率
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def main():
    """
    メインの実行フローを定義する関数です。
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset_dir = '/path/to/dataset'
    train_dataset, val_dataset, test_dataset = load_or_split_dataset(dataset_dir, data_transforms)

    class_names = train_dataset.classes

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter()

    best_val_acc = 0.0

    for epoch in range(10):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

    writer.close()

    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)

    test_preds = []
    test_labels = []

    for inputs, labels in tqdm(test_dataloader, desc="Predicting"):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

    class_accuracies = calculate_class_accuracies(test_preds, test_labels, class_names)
    precision = precision_score(test_labels, test_preds, average='weighted')
    recall = recall_score(test_labels, test_preds, average='weighted')
    f1 = f1_score(test_labels, test_preds, average='weighted')

    print("Class Accuracies:")
    for class_name, accuracy in class_accuracies.items():
        print(f"{class_name}: {accuracy * 100}%")

    print(f"Precision: {precision * 100}%")
    print(f"Recall: {recall * 100}%")
    print(f"F1 Score: {f1 * 100}%")

if __name__ == '__main__':
    main()
