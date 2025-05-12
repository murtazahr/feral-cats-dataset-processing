import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from networks.wide_restnet import Wide_ResNet


def main():
    data_dir = 'processed_dataset_obj_detection'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.1
    num_workers = 4
    dropout_rate = 0.3
    depth = 28
    widen_factor = 10
    log_interval = 10  # Log every 10 batches

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_dataset.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wide_ResNet(depth, widen_factor, dropout_rate, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.2)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Progress logging
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = running_loss / total
                acc = 100. * correct / total
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} Acc: {acc:.2f}%", flush=True)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] FINAL Loss: {running_loss/total:.4f} Acc: {100.*correct/total:.2f}%", flush=True)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        print(f"Validation Loss: {val_loss/val_total:.4f} Acc: {100.*val_correct/val_total:.2f}%", flush=True)

    torch.save(model.state_dict(), 'wideresnet28_10_custom.pth')
    print("Training complete. Model saved as wideresnet28_10_custom.pth")

if __name__ == "__main__":
    main()
