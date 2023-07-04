import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


def save_split_images(dataset, split_dataset, split_name):
    root_dir = dataset.root
    split_dir = os.path.join(os.path.dirname(root_dir), split_name)
    os.makedirs(split_dir, exist_ok=True)

    for idx in split_dataset.indices:
        img_path = dataset.imgs[idx][0]
        class_name = dataset.classes[dataset.imgs[idx][1]]
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img_file_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(class_dir, img_file_name))

def load_or_split_dataset(root_dir, data_transforms):
    output_dir = 'C:/Users/user/Desktop/vgg16model/VGG16/dataset'
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')

    # if os.path.isdir(output_dir):
    #     shutil.rmtree(output_dir)
    #
    # dataset = datasets.ImageFolder(root_dir, transform=data_transforms['train'])
    # train_size = int(0.8 * len(dataset))
    # val_size = int(0.1 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    #
    # save_split_images(dataset, train_dataset, os.path.join(output_dir, 'train'))
    # save_split_images(dataset, val_dataset, os.path.join(output_dir, 'val'))
    # save_split_images(dataset, test_dataset, os.path.join(output_dir, 'test'))

    # return train_dataset, val_dataset, test_dataset

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['val'])
    return train_dataset, val_dataset, test_dataset

def calculate_class_accuracies(predicted_labels, true_labels, class_names):
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

def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(224),
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

    dataset_dir = 'C:/Users/user/Desktop/vgg16model/VGG16/augment'
    train_dataset, val_dataset, test_dataset = load_or_split_dataset(dataset_dir, data_transforms)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

    dataloaders = {'train': train_loader, 'val': val_loader}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # num_classes = len(train_dataset.dataset.classes)
    num_classes = len(train_dataset.classes)
    vgg_net = models.vgg16(pretrained=False)
    # vgg_net.classifier[6] = nn.Linear(4096, num_classes)
    # vgg_net.classifier[6] = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(4096, num_classes)
    # )



    vgg_net = vgg_net.to(device)

    vgg_net.classifier = nn.Sequential(
        nn.Linear(2500, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(3000, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(3000, num_classes)
    )


    # vgg_net = models.resnet18(pretrained=False)
    # num_ftrs = vgg_net.fc.in_features
    # vgg_net.fc = nn.Linear(num_ftrs, num_classes)

    vgg_net = vgg_net.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(vgg_net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(vgg_net.parameters(), lr=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 40
    best_model_weights = vgg_net.state_dict()
    best_acc = 0.0

    writer = SummaryWriter('C:/Users/user/Desktop/vgg16model/VGG16/vgg_result/rice_disease_classification')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        vgg_net.train()
        running_loss = 0.0
        running_corrects = 0

        train_bar = tqdm(train_loader)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = vgg_net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader)  # 计算平均损失，除以训练步骤数
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        writer.add_scalar('train loss', epoch_loss, global_step=epoch)
        writer.add_scalar('train accuracy', epoch_acc, global_step=epoch)

        vgg_net.eval()
        running_loss = 0.0
        running_corrects = 0


        val_bar = tqdm(val_loader)
        for inputs, labels in val_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = vgg_net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = running_corrects.double() / len(val_dataset)
        print('Val Loss: {:.4f} Acc: {:.4f} num: %d'.format(epoch_loss, epoch_acc, len(val_dataset)))

        writer.add_scalar('val loss', epoch_loss, global_step=epoch)
        writer.add_scalar('val accuracy', epoch_acc, global_step=epoch)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = vgg_net.state_dict()

        # scheduler.step()

    writer.close()

    print('Best val Acc: {:4f}'.format(best_acc))

    vgg_net.load_state_dict(best_model_weights)
    torch.save(vgg_net.state_dict(), 'C:/Users/user/Desktop/vgg16model/VGG16/vgg_result/logs/rice_disease_vggnet.pth')

    vgg_net.eval()
    predicted_labels = []
    true_labels = []

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = vgg_net(inputs)
            _, preds = torch.max(outputs, 1)

        predicted_labels.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    class_accuracies = calculate_class_accuracies(predicted_labels, true_labels, train_dataset.dataset.classes)
    print("\n各種類の精度:")
    for class_name, accuracy in class_accuracies.items():
        print("  {}: {:.4f}".format(class_name, accuracy))

    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    print("\n評価指標:")
    print("  Precision: {:.4f}".format(precision))
    print("  Recall: {:.4f}".format(recall))
    print("  F1-score: {:.4f}".format(f1))

if __name__ == '__main__':
    main()
