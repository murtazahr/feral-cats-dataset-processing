import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import time
import datetime
import logging
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

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
    batch_size = 32  # Increased batch size for GPU
    num_epochs = 25
    learning_rate = 0.01
    num_workers = 4
    dropout_rate = 0.3
    depth = 28
    widen_factor = 10
    log_interval = 10  # Log every 10 batches

    # GPU optimization parameters
    use_amp = True  # Use Automatic Mixed Precision
    pin_memory = True  # Pin memory for faster GPU transfer
    cudnn_benchmark = True  # Enable cudnn benchmark for optimized performance
    prefetch_factor = 2  # Prefetch batches (only works with num_workers > 0)

    # Set GPU optimization flags
    if cudnn_benchmark and torch.cuda.is_available():
        cudnn.benchmark = True  # Set cudnn to benchmark mode for optimized performance
        logger.info("cuDNN benchmark enabled")
    else:
        logger.info("cuDNN benchmark disabled")

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
    logger.info(f"  AMP (mixed precision): {use_amp}")
    logger.info(f"  Pin memory: {pin_memory}")
    logger.info(f"  Data loader workers: {num_workers}")

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

    # Create data loaders with optimized settings for GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,  # Faster CPU to GPU transfers
        prefetch_factor=prefetch_factor if num_workers > 0 else None,  # Prefetch batches
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        drop_last=True  # Drop last incomplete batch for better GPU utilization
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Can use larger batch size for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )

    # Print dataset statistics
    num_classes = len(train_dataset.get_class_names())
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of test samples: {len(test_dataset)}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {train_dataset.get_class_names()}")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Log GPU information and optimize settings
    if device.type == 'cuda':
        # Log GPU details
        logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"  CUDA capability: {torch.cuda.get_device_capability(0)}")

        # Optimize memory allocations
        torch.cuda.empty_cache()

        # Run a warmup to initialize CUDA context (reduces timing noise in first batches)
        logger.info("Running CUDA warmup...")
        dummy_input = torch.zeros(1, 3, 32, 32, device=device)
        dummy_model = nn.Conv2d(3, 3, 3).to(device)
        for _ in range(3):  # A few warmup iterations
            _ = dummy_model(dummy_input)
        torch.cuda.synchronize()
        logger.info("CUDA warmup completed")

    # Initialize model
    logger.info(f"Initializing Wide ResNet-{depth}-{widen_factor} model...")
    model = Wide_ResNet(depth, widen_factor, dropout_rate, num_classes).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of model parameters: {total_params/1e6:.2f}M (trainable: {trainable_params/1e6:.2f}M)")

    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        model = nn.DataParallel(model)

    # Loss and optimizer with optimized settings
    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.2)

    # Initialize mixed precision training if available and selected
    scaler = amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    if use_amp and device.type == 'cuda':
        logger.info("Using automatic mixed precision (FP16) training")
    else:
        logger.info("Using full precision (FP32) training")

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

        # Move these outside the batch loop for better performance
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device (asynchronously if pin_memory=True)
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Mixed precision forward pass
            with amp.autocast(enabled=use_amp and device.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Optimizer step with gradient scaling for mixed precision
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Ensure all asynchronous CUDA operations are done before moving on
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.synchronize()

            # Update metrics
            running_loss += loss.item() * images.size(0)
            with torch.no_grad():  # Save memory during inference
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

                # Get GPU memory usage
                if device.type == 'cuda':
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
                    memory_info = f"GPU Mem: {gpu_memory_allocated:.2f}/{gpu_memory_reserved:.2f} GB"
                else:
                    memory_info = ""

                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Step [{global_step}] "
                    f"Loss: {avg_loss:.4f} "
                    f"Acc: {acc:.2f}% "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                    f"Time: {batch_time:.3f}s "
                    f"Img/s: {images_per_sec:.1f} "
                    f"{memory_info} "
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
        with torch.no_grad(), amp.autocast(enabled=use_amp and device.type == 'cuda'):
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # Forward pass with mixed precision
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

        # Synchronize GPU operations before measuring time
        if device.type == 'cuda':
            torch.cuda.synchronize()

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
    if isinstance(model, nn.DataParallel):
        # Save the model state without DataParallel wrapper
        torch.save(model.module.state_dict(), 'final_feral_cats_model.pth')
    else:
        torch.save(model.state_dict(), 'final_feral_cats_model.pth')

    # Calculate final training time
    total_training_time = time.time() - start_training_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Log final results
    logger.info("=" * 80)
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    logger.info(f"Final model saved as 'final_feral_cats_model.pth'")
    logger.info(f"Best model saved as 'best_feral_cats_model.pth'")
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.2f}s")
    logger.info(f"Average batch time: {sum(batch_times)/len(batch_times):.4f}s")

    # GPU memory statistics if available
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")

    logger.info("=" * 80)

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, num_epochs)

    # Load the best model for evaluation
    logger.info("Loading best model for detailed evaluation...")
    if isinstance(model, nn.DataParallel):
        best_model = Wide_ResNet(depth, widen_factor, dropout_rate, num_classes).to(device)
        best_model.load_state_dict(torch.load('best_feral_cats_model.pth'))
        if torch.cuda.device_count() > 1:
            best_model = nn.DataParallel(best_model)
    else:
        best_model = model
        best_model.load_state_dict(torch.load('best_feral_cats_model.pth'))

    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(best_model, test_loader, device, num_classes)

    # Log all metrics
    log_detailed_metrics(detailed_metrics, train_dataset.get_class_names(), logger)

    # Plot confusion matrix and per-class metrics
    plot_confusion_matrix(detailed_metrics, train_dataset.get_class_names())


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

def calculate_detailed_metrics(model, test_loader, device, num_classes):
    """
    Calculate detailed metrics after training is complete

    Returns:
        Dictionary containing all metrics and confusion matrix
    """
    model.eval()

    # Initialize storage for predictions and targets
    all_targets = []
    all_predictions = []
    all_scores = []  # For storing raw prediction scores (for top-k accuracy)

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, predictions = outputs.max(1)

            # Store raw scores for top-k accuracy calculation
            all_scores.append(outputs.cpu())

            # Store targets and predictions
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Convert lists to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_scores = torch.cat(all_scores, dim=0).numpy()

    # Calculate metrics
    metrics = {}

    # Overall accuracy
    metrics['accuracy'] = (all_targets == all_predictions).mean() * 100

    # Top-5 accuracy (if num_classes >= 5)
    if num_classes >= 5:
        top5_count = 0
        for i in range(len(all_targets)):
            # Get indices of top 5 predictions for this sample
            top5_indices = np.argsort(all_scores[i])[-5:]
            if all_targets[i] in top5_indices:
                top5_count += 1
        metrics['top5_accuracy'] = (top5_count / len(all_targets)) * 100
    else:
        metrics['top5_accuracy'] = 'N/A (less than 5 classes)'

    # Per-class and micro/macro/weighted metrics
    metrics['per_class_precision'] = precision_score(all_targets, all_predictions,
                                                     average=None,
                                                     zero_division=0)
    metrics['per_class_recall'] = recall_score(all_targets, all_predictions,
                                               average=None,
                                               zero_division=0)
    metrics['per_class_f1'] = f1_score(all_targets, all_predictions,
                                       average=None,
                                       zero_division=0)

    # Micro average (calculate metrics globally by counting total TP, FN and FP)
    metrics['micro_precision'] = precision_score(all_targets, all_predictions,
                                                 average='micro',
                                                 zero_division=0)
    metrics['micro_recall'] = recall_score(all_targets, all_predictions,
                                           average='micro',
                                           zero_division=0)
    metrics['micro_f1'] = f1_score(all_targets, all_predictions,
                                   average='micro',
                                   zero_division=0)

    # Macro average (calculate metrics for each class and take unweighted mean)
    metrics['macro_precision'] = precision_score(all_targets, all_predictions,
                                                 average='macro',
                                                 zero_division=0)
    metrics['macro_recall'] = recall_score(all_targets, all_predictions,
                                           average='macro',
                                           zero_division=0)
    metrics['macro_f1'] = f1_score(all_targets, all_predictions,
                                   average='macro',
                                   zero_division=0)

    # Weighted average (calculate metrics for each class and take mean weighted by support)
    metrics['weighted_precision'] = precision_score(all_targets, all_predictions,
                                                    average='weighted',
                                                    zero_division=0)
    metrics['weighted_recall'] = recall_score(all_targets, all_predictions,
                                              average='weighted',
                                              zero_division=0)
    metrics['weighted_f1'] = f1_score(all_targets, all_predictions,
                                      average='weighted',
                                      zero_division=0)

    # Generate confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(all_targets, all_predictions)

    return metrics


def log_detailed_metrics(metrics, class_names, logger):
    """
    Log detailed metrics to console and file via logger
    """
    logger.info("=" * 80)
    logger.info("DETAILED EVALUATION METRICS")
    logger.info("=" * 80)

    # Log top-1 and top-5 accuracy
    logger.info(f"Top-1 Accuracy: {metrics['accuracy']:.2f}%")
    if isinstance(metrics['top5_accuracy'], str):
        logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']}")
    else:
        logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")

    # Log micro/macro/weighted averages
    logger.info("\nAggregated Metrics:")
    logger.info(f"  Micro-average Precision: {metrics['micro_precision']:.4f}")
    logger.info(f"  Micro-average Recall: {metrics['micro_recall']:.4f}")
    logger.info(f"  Micro-average F1: {metrics['micro_f1']:.4f}")
    logger.info(f"  Macro-average Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"  Macro-average Recall: {metrics['macro_recall']:.4f}")
    logger.info(f"  Macro-average F1: {metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted-average Precision: {metrics['weighted_precision']:.4f}")
    logger.info(f"  Weighted-average Recall: {metrics['weighted_recall']:.4f}")
    logger.info(f"  Weighted-average F1: {metrics['weighted_f1']:.4f}")

    # Log per-class metrics
    logger.info("\nPer-class Metrics:")
    for i, class_name in enumerate(class_names):
        logger.info(f"  Class {i} ({class_name}):")
        logger.info(f"    Precision: {metrics['per_class_precision'][i]:.4f}")
        logger.info(f"    Recall: {metrics['per_class_recall'][i]:.4f}")
        logger.info(f"    F1-score: {metrics['per_class_f1'][i]:.4f}")

    logger.info("=" * 80)


def plot_confusion_matrix(metrics, class_names):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    conf_matrix = metrics['confusion_matrix']

    # Create dataframe for better visualization
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

    # Get percentage of examples in each class for normalization
    row_sums = conf_matrix.sum(axis=1)
    normalized_cm = conf_matrix / row_sums[:, np.newaxis]
    df_cm_norm = pd.DataFrame(normalized_cm, index=class_names, columns=class_names)

    # Plot normalized confusion matrix
    sns.heatmap(df_cm_norm, annot=True, cmap="Blues", fmt='.2f', cbar=True, vmin=0, vmax=1)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()

    # Create directory for plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig('plots/confusion_matrix.png')
    logging.info("Confusion matrix saved to 'plots/confusion_matrix.png'")

    # Create more detailed metrics plot
    plt.figure(figsize=(15, 10))

    # Prepare metrics for plotting
    class_metrics = pd.DataFrame({
        'Precision': metrics['per_class_precision'],
        'Recall': metrics['per_class_recall'],
        'F1-Score': metrics['per_class_f1']
    }, index=class_names)

    # Plot per-class metrics
    ax = class_metrics.plot(kind='bar', figsize=(15, 7))
    plt.title('Precision, Recall and F1-Score per Class')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.ylim([0, 1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('plots/per_class_metrics.png')
    logging.info("Per-class metrics plot saved to 'plots/per_class_metrics.png'")


if __name__ == "__main__":
    main()
