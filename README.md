#   HIL-MDF: Human-In-The-Loop Multimodal Deepfake Detection

[![Paper](https://img.shields.io/badge/Paper-IEEE%203SCEA2026-blue)](https://github.com/H-Beheiry/Multimodal-Deepfake-Detection-with-XAI)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

HIL-MDF is a multimodal deepfake detection framework designed to provide both high accuracy and transparency. By integrating audio and visual streams with Explainable AI (XAI) methods, the system generates interpretable evidence to support human verification in digital forensic investigations.

## Key Features

- **Multimodal Fusion:** The architecture utilizes a mid-level fusion strategy to combine spatiotemporal visual embeddings with audio spectral features.
    
- **Explainable AI (XAI):** The system implements three distinct techniques—Saliency Maps, Integrated Gradients, and Layer-Grad-CAM—to provide visual reasoning for model predictions.
    
- **Human-In-The-Loop (HITL):** An interactive decision support interface allows users to cross-reference media with model explanations and verify findings.
    
- **Superior Performance:** The system achieved an accuracy of 89.75% on the FakeAVCeleb dataset, significantly outperforming unimodal baselines.
    

---

## System Architecture

The framework consists of a two-layer structure: the User Interaction Layer and the Multimodal Deepfake Detection and Learning Engine.

### 1. Detection Pipeline

The engine follows a sequential pipeline including data preprocessing, feature extraction, and modality fusion.

- **Visual Branch:** Uses an EfficientNet-B0 backbone coupled with 1D convolution to capture spatiotemporal inconsistencies.
    
- **Audio Branch:** Employs a specialized 2D Convolutional Neural Network (CNN) to process Mel-Spectrograms.
    

### 2. Explainability Generation

Attribution maps are generated to highlight specific pixels or temporal segments that influenced the classification. For audio, the system performs a temporal analysis to transform attribution maps into significant intervals displayed on a waveform.

---

## Experimental Results

The model was evaluated using the FakeAVCeleb dataset, which contains both real and manipulated audio-visual content.

|**Modality**|**Accuracy**|**Precision**|**Recall**|**F1-Score**|
|---|---|---|---|---|
|Visual Only|0.6950|0.9784|0.6067|0.7487|
|Auditory Only|0.8025|0.9911|0.7433|0.8495|
|**Audio-Visual (HIL-MDF)**|**0.8975**|**0.9778**|**0.8833**|**0.9282**|

---

## Installation and Usage

### Prerequisites

- Python 3.8+
    
- PyTorch

- Captum
    
- NVIDIA GPU
    

### Setup

Bash

```
git clone https://github.com/H-Beheiry/Multimodal-Deepfake-Detection-with-XAI.git
cd Multimodal-Deepfake-Detection-with-XAI
```

---

## Citation

If you use this work in your research, please cite:

Code snippet

```
@inproceedings{elbeheiry2026hilmdf,
  title={HIL-MDF: A Human-in-the-Loop Multimodal Deepfake Detection System},
  author={El-Beheiry, Hamza and Abdulwagid, Mahmoud and Asmah, Ahmed and Moussa, Sherin and Zaher, Omar},
  booktitle={IEEE International Conference on Smart Systems, Control, and Engineering Applications (IEEE 3SCEA2026)},
  year={2026}
}
```

---

**Affiliation:** Laboratoire Interdisciplinaire de l'Université Française d'Égypte (UFEID Lab).
