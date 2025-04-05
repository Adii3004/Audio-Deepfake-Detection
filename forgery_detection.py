# -*- coding: utf-8 -*-
"""
# Audio Deepfake Detection System

A Hybrid Approach with Fine-tuned Wav2Vec 2.0 and CRNN Architecture

This implementation fine-tunes Wav2Vec 2.0 and combines it with a
lightweight Convolutional Recurrent Neural Network (CRNN) for efficient
and accurate real-time deepfake detection using the ASVspoof 2019 dataset.
"""

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
# from transformers import AdamW, get_linear_schedule_with_warmup
import kagglehub
import librosa
import librosa.display
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import time

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

"""## 1. Dataset Preparation"""

# Download dataset using kagglehub
path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
print("Dataset downloaded to:", path)

# Define paths
LA_TRAIN_PATH = os.path.join(path, "LA/LA/ASVspoof2019_LA_train/flac/")
LA_DEV_PATH = os.path.join(path, "LA/LA/ASVspoof2019_LA_dev/flac/")
LA_EVAL_PATH = os.path.join(path, "LA/LA/ASVspoof2019_LA_eval/flac/")
LA_TRAIN_PROTOCOL = os.path.join(path, "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
LA_DEV_PROTOCOL = os.path.join(path, "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
LA_EVAL_PROTOCOL = os.path.join(path, "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")

# Verify dataset structure
print(f"Train audio files: {len(os.listdir(LA_TRAIN_PATH))}")
print(f"Development audio files: {len(os.listdir(LA_DEV_PATH))}")
print(f"Evaluation audio files: {len(os.listdir(LA_EVAL_PATH))}")

"""## 2. Custom Audio Dataset"""

class ASVSpoofDataset(Dataset):
    """
    Custom dataset for the ASVspoof dataset that loads audio on-the-fly
    and applies preprocessing
    """
    def __init__(self, audio_dir, protocol_file, feature_extractor, max_length=160000):
        """
        Initialize the dataset

        Args:
            audio_dir: Directory containing audio files
            protocol_file: File with labels and metadata
            feature_extractor: Wav2Vec2 feature extractor
            max_length: Maximum length of audio in samples (10s at 16kHz)
        """
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.feature_extractor = feature_extractor

        # Read protocol file
        self.protocols = pd.read_csv(protocol_file, sep=' ', header=None,
                                   names=['speaker_id', 'file_id', 'dash1', 'dash2', 'target'])

        # Create file paths and labels
        self.file_paths = [os.path.join(audio_dir, row["file_id"] + ".flac")
                           for _, row in self.protocols.iterrows()]
        self.labels = [1 if target == "bonafide" else 0
                       for target in self.protocols["target"]]

        # Filter out non-existent files
        valid_indices = [i for i, path in enumerate(self.file_paths) if os.path.exists(path)]
        self.file_paths = [self.file_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load and preprocess audio
        waveform, sample_rate = self.preprocess_audio(file_path)

        # Convert to input features
        input_values = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.squeeze(0)

        return input_values, label

    def preprocess_audio(self, file_path):
        """
        Load and preprocess audio file

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Normalize waveform
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)

        # Pad or truncate to max_length
        if waveform.shape[1] < self.max_length:
            # Pad with zeros
            padding = torch.zeros(1, self.max_length - waveform.shape[1])
            waveform = torch.cat((waveform, padding), dim=1)
        elif waveform.shape[1] > self.max_length:
            # Truncate
            waveform = waveform[:, :self.max_length]

        return waveform.squeeze(0), sample_rate

"""## 3. Audio Visualization Functions"""

def visualize_audio(waveform, sample_rate, title="Waveform"):
    """
    Visualize audio waveform and spectrogram

    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))

    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.title(f"{title} - Waveform")
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    librosa.display.waveshow(waveform, sr=sample_rate)

    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.title(f"{title} - Mel Spectrogram")
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max),
                            y_axis='mel', x_axis='time', sr=sample_rate)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

"""## 4. Model Architecture"""

class Wav2VecCRNN(nn.Module):
    """
    Hybrid model that fine-tunes Wav2Vec 2.0 and combines it with a CRNN
    """
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base", num_classes=2):
        super(Wav2VecCRNN, self).__init__()

        # Load pretrained Wav2Vec 2.0 model
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        wav2vec_config = Wav2Vec2Config.from_pretrained(pretrained_model_name)

        # Get output dimension from config
        self.hidden_size = wav2vec_config.hidden_size  # Typically 768

        # Convolutional layers
        self.conv1 = nn.Conv1d(self.hidden_size, 256, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        # Recurrent layer
        self.gru = nn.GRU(128, 64, bidirectional=True, batch_first=True)

        # Attention mechanism
        self.attention = nn.Linear(128, 1)  # Bidirectional GRU gives 128 features

        # Classification layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, attention_mask=None):
        """
        Forward pass through the network

        Args:
            x: Input audio tensor
            attention_mask: Optional attention mask

        Returns:
            Classification logits
        """
        # Pass through Wav2Vec 2.0
        wav2vec_outputs = self.wav2vec(x, attention_mask=attention_mask)
        hidden_states = wav2vec_outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Transpose for convolutional layers [batch, hidden, seq_len]
        conv_input = hidden_states.transpose(1, 2)

        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(conv_input)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Transpose back for GRU [batch, seq_len, features]
        x = x.transpose(1, 2)

        # GRU layer
        x, _ = self.gru(x)  # [batch, seq_len, 2*64]

        # Attention mechanism
        attention_weights = F.softmax(self.attention(x), dim=1)  # [batch, seq_len, 1]
        x = torch.sum(x * attention_weights, dim=1)  # [batch, 128]

        # Classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def freeze_feature_extractor(self):
        """Freeze the Wav2Vec feature extractor layers"""
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True

"""## 5. Collate Function and Data Preparation"""

def collate_fn(batch):
    """
    Custom collate function to handle variable length inputs

    Args:
        batch: List of (input_values, label) tuples

    Returns:
        Batched inputs and labels
    """
    input_values = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    # Pad inputs to the same length
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

    # Create attention mask
    attention_mask = torch.ones_like(input_values).to(torch.long)
    for i, item in enumerate(batch):
        attention_mask[i, item[0].shape[0]:] = 0

    return input_values, attention_mask, labels

# Load feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Create datasets
train_dataset = ASVSpoofDataset(LA_TRAIN_PATH, LA_TRAIN_PROTOCOL, feature_extractor)
dev_dataset = ASVSpoofDataset(LA_DEV_PATH, LA_DEV_PROTOCOL, feature_extractor)
test_dataset = ASVSpoofDataset(LA_EVAL_PATH, LA_EVAL_PROTOCOL, feature_extractor)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4
)

print(f"Training samples: {len(train_dataset)}")
print(f"Development samples: {len(dev_dataset)}")
print(f"Test samples: {len(test_dataset)}")

"""## 6. Training Functions"""

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Train for one epoch

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to train on

    Returns:
        Average loss and accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, attention_mask, labels in progress_bar:
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })

    return total_loss / len(dataloader), 100 * correct / total

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model

    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Average loss, accuracy, predictions, and labels
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1]  # Probability of bonafide
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / len(dataloader), 100 * correct / total, all_probs, all_labels

"""## 7. Training Loop"""

def train_model(model, train_loader, dev_loader, num_epochs=50):
    """
    Train the model

    Args:
        model: Model to train
        train_loader: Training data loader
        dev_loader: Validation data loader
        num_epochs: Number of epochs to train

    Returns:
        Trained model
    """
    # Set up optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    # Total number of training steps
    total_steps = len(train_loader) * num_epochs

    # Create scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Phase 1: Freeze feature extractor for initial epochs
        if epoch < 3:
            model.freeze_feature_extractor()
            print("Feature extractor frozen")
        else:
            model.unfreeze_all()
            print("All layers unfrozen for fine-tuning")

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Evaluate on validation set
        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, dev_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Calculate EER
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]
        print(f"Equal Error Rate (EER): {eer:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with accuracy: {best_val_acc:.2f}%")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'eer': eer
            }, "best_wav2vec_crnn_model.pt")

    # Load best model
    model.load_state_dict(best_model_state)
    return model

"""## 8. Evaluation Functions"""

def detailed_evaluation(model, dataloader, device):
    """
    Perform detailed evaluation with various metrics

    Args:
        model: Model to evaluate
        dataloader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, attention_mask, labels in tqdm(dataloader, desc="Detailed Evaluation"):
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Bonafide probability

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Find EER (Equal Error Rate)
    eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
    eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]

    # Binary predictions at EER threshold
    binary_preds = (all_probs >= eer_threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)

    # Classification report
    report = classification_report(
        all_labels, binary_preds, target_names=['Spoof', 'Bonafide'], output_dict=True
    )

    return {
        'roc_auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'confusion_matrix': cm,
        'classification_report': report,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'predictions': binary_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def plot_evaluation_results(metrics):
    """
    Plot evaluation results

    Args:
        metrics: Dictionary with evaluation metrics
    """
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(
        metrics['fpr'],
        metrics['tpr'],
        color='darkorange',
        lw=2,
        label=f"ROC curve (AUC = {metrics['roc_auc']:.3f})"
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Mark EER point
    eer_index = np.argmin(np.abs(metrics['fpr'] - (1 - metrics['tpr'])))
    plt.plot(
        metrics['fpr'][eer_index],
        metrics['tpr'][eer_index],
        'ro',
        markersize=8,
        label=f"EER = {metrics['eer']:.3f}"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Spoof', 'Bonafide']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = metrics['confusion_matrix'].max() / 2.
    for i in range(metrics['confusion_matrix'].shape[0]):
        for j in range(metrics['confusion_matrix'].shape[1]):
            plt.text(
                j, i,
                format(metrics['confusion_matrix'][i, j], 'd'),
                horizontalalignment="center",
                color="white" if metrics['confusion_matrix'][i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Print detailed metrics
    report = metrics['classification_report']
    print("\nClassification Report:")
    print(f"Spoof - Precision: {report['Spoof']['precision']:.3f}, Recall: {report['Spoof']['recall']:.3f}, F1: {report['Spoof']['f1-score']:.3f}")
    print(f"Bonafide - Precision: {report['Bonafide']['precision']:.3f}, Recall: {report['Bonafide']['recall']:.3f}, F1: {report['Bonafide']['f1-score']:.3f}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Equal Error Rate (EER): {metrics['eer']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

"""## 9. Real-time Inference"""

def process_audio_realtime(audio_file, model, feature_extractor, device, threshold=None):
    """
    Process audio in real-time simulation

    Args:
        audio_file: Path to audio file
        model: Trained model
        feature_extractor: Feature extractor
        device: Device to run inference on
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary with predictions and processing time
    """
    start_time = time.time()

    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_file)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Normalize
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)

    # Extract features
    input_values = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    input_values = input_values.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_values)
        prob = F.softmax(outputs, dim=1)[0, 1].item()  # Probability of bonafide

    # Use default threshold if none provided
    if threshold is None:
        threshold = 0.5

    # End timing
    process_time = time.time() - start_time

    return {
        'bonafide_probability': prob,
        'prediction': 'Bonafide' if prob >= threshold else 'Spoof',
        'processing_time': process_time
    }

"""## 10. Main Training and Evaluation"""

def main():
    """
    Main function to train and evaluate the model
    """
    print("Initializing Wav2Vec2-CRNN hybrid model...")
    model = Wav2VecCRNN().to(device)

    print("Starting model training...")
    model = train_model(model, train_loader, dev_loader, num_epochs=10)

    print("Performing detailed evaluation...")
    test_metrics = detailed_evaluation(model, test_loader, device)
    plot_evaluation_results(test_metrics)

    print("Testing real-time inference...")
    # Test on a sample file
    sample_file = test_dataset.file_paths[0]
    result = process_audio_realtime(
        sample_file,
        model,
        feature_extractor,
        device,
        threshold=test_metrics['eer_threshold']
    )

    print(f"File: {os.path.basename(sample_file)}")
    print(f"True label: {'Bonafide' if test_dataset.labels[0] == 1 else 'Spoof'}")
    print(f"Prediction: {result['prediction']}")
    print(f"Bonafide Probability: {result['bonafide_probability']:.4f}")
    print(f"Processing Time: {result['processing_time']:.4f} seconds")

    print("Saving model for deployment...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'eer_threshold': test_metrics['eer_threshold'],
        'feature_extractor': feature_extractor.__dict__,
        'architecture': 'Wav2VecCRNN'
    }, "wav2vec_crnn_deepfake_detector.pt")

    print("Done!")

if __name__ == "__main__":
    main()

