import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import time
import datetime
import logging
from dotenv import load_dotenv

from networks.wide_restnet import Wide_ResNet


class FeralCatsDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.get_class_names())}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract pixel values and reshape to 32x32x3
        pixel_values = []
        for i in range(3072):  # 32x32x3 = 3072
            pixel_values.append(item[f'pixel_{i}'])

        # Convert to numpy array and reshape to 32x32x3
        image = np.array(pixel_values, dtype=np.uint8).reshape(32, 32, 3)

        # Convert label to integer
        label = item['label']

        # Apply transformations if specified
        if self.transform:
            # Convert numpy array to PIL Image
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # Convert from HWC to CHW format

        return image, label

    def get_class_names(self):
        # Get unique class names from dataset
        classes = {}
        for item in self.dataset:
            class_name = item['class_name']
            label = item['label']
            classes[label] = class_name

        # Sort by label to ensure consistent ordering
        return [classes[i] for i in range(len(classes))]


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Dataset parameters
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.1
    num_workers = 4
    dropout_rate = 0.3
    depth = 28
    widen_factor = 10
    log_interval = 10  # Log every 10 batches

    # Load environment variables from .env file
    load_dotenv()

    # Get Hugging Face token from environment variable
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.warning("No Hugging Face token found in environment. If the dataset is private, this will fail.")
        logger.warning("Create a .env file with your HF_TOKEN or set it as an environment variable.")
    else:
        logger.info("Hugging Face token loaded successfully")

    # Log training parameters
    logger.info(f"Training configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num epochs: {num_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Dropout rate: {dropout_rate}")
    logger.info(f"  Model depth: {depth}")
    logger.info(f"  Model widen factor: {widen_factor}")

    # Load dataset from Hugging Face
    logger.info("Loading dataset from Hugging Face...")
    start_time = time.time()
    dataset = load_dataset("murtazahr/feral-cats", token=hf_token)
    logger.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create custom datasets
    logger.info("Creating datasets and dataloaders...")
    train_dataset = FeralCatsDataset(dataset['train'], transform=transform_train)
    test_dataset = FeralCatsDataset(dataset['test'], transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Print dataset statistics
    num_classes = len(train_dataset.get_class_names())
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of test samples: {len(test_dataset)}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {train_dataset.get_class_names()}")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize model
    logger.info(f"Initializing Wide ResNet-{depth}-{widen_factor} model...")
    model = Wide_ResNet(depth, widen_factor, dropout_rate, num_classes).to(device)
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.2)

    # Training metrics tracking
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Initialize timing metrics
    total_train_time = 0
    epoch_times = []
    batch_times = []

    # Global step counter for logging
    global_step = 0

    # Training loop
    best_acc = 0.0
    start_training_time = time.time()
    logger.info("Starting training...")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Initialize batch timing
        batch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Calculate batch processing time
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)

            # Update global step
            global_step += 1

            # Granular progress logging
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = running_loss / total
                acc = 100. * correct / total
                avg_batch_time = sum(batch_times[-log_interval:]) / min(log_interval, len(batch_times[-log_interval:]))
                images_per_sec = batch_size / avg_batch_time

                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Step [{global_step}] "
                    f"Loss: {avg_loss:.4f} "
                    f"Acc: {acc:.2f}% "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                    f"Time: {batch_time:.3f}s "
                    f"Img/s: {images_per_sec:.1f} "
                    f"ETA: {datetime.timedelta(seconds=int(avg_batch_time * (len(train_loader) - batch_idx - 1)))}"
                )

            # Reset batch timing
            batch_start = time.time()

        # Update learning rate
        scheduler.step()

        # Calculate epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        total_train_time += epoch_time

        # Save epoch metrics
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] COMPLETE - "
            f"Training Loss: {epoch_loss:.4f} "
            f"Acc: {epoch_acc:.2f}% "
            f"Time: {epoch_time:.2f}s "
            f"ETA: {datetime.timedelta(seconds=int((num_epochs - epoch - 1) * sum(epoch_times) / len(epoch_times)))}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_start = time.time()
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        logger.info("Starting validation...")
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1

        # Calculate validation metrics
        val_epoch_loss = val_loss / val_total
        val_accuracy = 100. * val_correct / val_total
        val_time = time.time() - val_start

        # Save validation metrics
        val_losses.append(val_epoch_loss)
        val_accs.append(val_accuracy)

        # Log validation results
        logger.info(
            f"Validation - "
            f"Loss: {val_epoch_loss:.4f} "
            f"Acc: {val_accuracy:.2f}% "
            f"Time: {val_time:.2f}s"
        )

        # Log per-class accuracy
        for i in range(num_classes):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                class_name = train_dataset.get_class_names()[i]
                logger.info(f"  Class {i} ({class_name}): {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

        # Save model if it's the best so far
        if val_accuracy > best_acc:
            logger.info(f"New best accuracy: {val_accuracy:.2f}% (previous: {best_acc:.2f}%)")
            best_acc = val_accuracy
            torch.save(model.state_dict(), 'best_feral_cats_model.pth')
            logger.info("Saved new best model")

        # Calculate expected time to finish training
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta = (num_epochs - epoch - 1) * avg_epoch_time
        logger.info(f"Expected time to finish training: {datetime.timedelta(seconds=int(eta))}")

        # Log separator for readability
        logger.info("-" * 80)

    # Save final model
    torch.save(model.state_dict(), 'final_feral_cats_model.pth')

    # Calculate final training time
    total_training_time = time.time() - start_training_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Log final results
    logger.info("=" * 80)
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    logger.info(f"Final model saved as 'final_feral_cats_model.pth'")
    logger.info(f"Best model saved as 'best_feral_cats_model.pth'")
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.2f}s")
    logger.info(f"Average batch time: {sum(batch_times)/len(batch_times):.4f}s")
    logger.info("=" * 80)

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, num_epochs)


def plot_training_history(train_losses, val_losses, train_accs, val_accs, num_epochs):
    """Plot and save training history"""
    epochs = range(1, len(train_losses) + 1)

    # Create directory for plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/training_history.png')

    logging.info("Training history plots saved to 'plots/training_history.png'")


def visualize_sample(dataset, idx):
    """Helper function to visualize a sample from the dataset"""
    image, label = dataset[idx]
    class_name = dataset.get_class_names()[label]

    # If image is a tensor, convert to numpy array
    if isinstance(image, torch.Tensor):
        # If normalized, approximately denormalize for visualization
        if image.max() <= 1.0:
            image = image.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
        else:
            image = image.permute(1, 2, 0).numpy() / 255.0

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(f"Class: {class_name} (Label: {label})")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
