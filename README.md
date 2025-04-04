# Audio Deepfake Detection

This project provides a solution for detecting manipulated audio using **transfer learning techniques**. The system is trained and evaluated on the **ASVspoof 2019 dataset**, which contains both **genuine speech recordings ("bonafide")** and **synthetic/manipulated speech ("spoof")**.

## Key Features

**Transfer Learning**: Fine-tunes **Facebook's Wav2Vec 2.0** pre-trained model for robust feature extraction.  
**Hybrid Architecture**: Combines **CNN layers** for spatial feature extraction with a **bidirectional GRU** for temporal analysis.  
**Attention Mechanism**: Enhances performance by focusing on the most discriminative parts of the audio signal.  
**Real-time Inference**: Optimized for efficient processing with **minimal latency**.  
**Comprehensive Evaluation**: Computes **Equal Error Rate (EER)**, **ROC curves**, and **confusion matrices** for performance assessment.  

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/audio-deepfake-detection.git
cd audio-deepfake-detection
```

### Create a Virtual Environment (Recommended)
``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dataset
This project uses the ASVspoof 2019 Logical Access (LA) dataset. The dataset will be automatically downloaded using kagglehub during the first run.

Setup Kaggle API Credentials
Create a Kaggle account if you don't have one.

Go to your Kaggle Account Settings and create an API token.

Download the kaggle.json file and place it in ~/.kaggle/.

### Model Architecture
The system uses a hybrid deep learning model:

- Wav2Vec 2.0 Base: Extracts robust speech representations.
- Convolutional Layers (CNN): Detects local patterns in the extracted embeddings.
- Bidirectional GRU: Captures temporal dependencies in both directions.
- Attention Mechanism: Focuses on the most important frames for classification.
- Fully Connected Layer: Outputs the final bonafide/spoof decision.

