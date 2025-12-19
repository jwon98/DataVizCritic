# Data Visualization Critic

**AI-Powered Statistical Code Review for Python**

Automatically detect statistical methodology errors and visualization issues in data analysis code using a fine-tuned Llama-3-8B model.

---

## 🎯 Project Overview

Data Visualization Critic is a specialized language model fine-tuned to review Python data analysis code and identify common statistical and visualization errors. It provides:

- **Error Detection**: Identifies 15+ types of statistical and visualization errors
- **Explanations**: Explains why errors are problematic and their real-world consequences  
- **Fixes**: Suggests concrete improvements and proper statistical practices
- **Interactive Demo**: Gradio web interface with live code execution and plot visualization

**Developed for:** COMS 4995 - Applied Machine Learning (Columbia University)

**Team:** Rencong Jiang, Charles Weber, John Won, and Amir Yaghoobi

---

## 🚀 Key Features

### Statistical Errors Detected (10 types)
- Correlation vs Causation Confusion
- Simpson's Paradox
- Multiple Testing without Correction
- P-hacking / Data Dredging
- Omitted Confounding Variables
- Survivorship Bias
- Regression to the Mean
- Base Rate Neglect
- Extrapolation Beyond Data Range
- Statistical Assumption Violations

### Visualization Errors Detected (5 types)
- Truncated Y-Axis Manipulation
- Misleading Dual Axes
- Inappropriate Chart Types
- Overplotting without Transparency
- Missing Uncertainty Visualization

---

## 📊 Results

**Model Performance:**
- **Training Data:** 738 examples (390 negative + 48 positive)
- **Detection Accuracy:** 100% on held-out test set (8/8 error types)
- **Training Loss:** 0.873 → 0.078 (91.1% reduction)
- **Model:** Llama-3-8B-Instruct with LoRA fine-tuning

**Example Detections:**
- ✅ Truncated axes making small changes appear massive
- ✅ Multiple testing without Bonferroni correction
- ✅ Simpson's Paradox from aggregating across confounders
- ✅ Correlation-causation confusion in observational data

---

## 🛠️ Technical Approach

### Architecture
- **Base Model:** Llama-3-8B-Instruct
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit (NF4) for memory efficiency
- **Training:** 3 epochs on 738 examples (~3.5 hours on T4 GPU)

### Pipeline
1. **Data Generation:** Synthetic examples using Llama-3-8B
2. **Fine-tuning:** LoRA with 4-bit quantization
3. **Evaluation:** Testing on held-out error types
4. **Deployment:** Gradio web interface with plot visualization

### Key Technologies
- PyTorch + HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Gradio for demo interface
- Google Colab for training (free T4 GPU)

---

## 📁 Repository Structure
```
dataviz-critic/
├── README.md
├── 1_data_generation.ipynb      # Phase 1: Generate training data
├── 2_model_training.ipynb       # Phase 2: LoRA fine-tuning
├── 3_demo_evaluation.ipynb      # Phase 3: Demo & evaluation
├── data/
│   ├── training_data_combined.csv    # Training dataset (sample)
│   └── evaluation_results.csv        # Test results
├── assets/
│   └── demo_screenshot.png           # Demo screenshot
└── requirements.txt
```

**Note:** The trained model weights (~20MB LoRA adapters) are stored separately due to size. See [Model Weights](#model-weights) section.

---

## 🚦 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (for inference)
- 16GB+ RAM recommended

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/[your-username]/dataviz-critic.git
cd dataviz-critic
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up HuggingFace token** (for Llama-3 access):
```bash
export HF_TOKEN="your_huggingface_token"
```

### Running the Demo

Open `3_demo_evaluation.ipynb` in Google Colab or Jupyter:

1. Run cells 1-2 to load the model
2. Run cells 7-10 to launch the Gradio interface
3. Access the demo via the public URL

**Try these examples:**
- Truncated axis manipulation
- Multiple testing without correction
- Simpson's Paradox

---

## 🎓 Model Training

### Training Data
- **738 total examples:**
  - 390 negative examples (flawed code with errors)
  - 48 positive examples (good code demonstrating best practices)
  - 300 examples from V1 dataset
- **Coverage:** 15 error types across 4 domains (healthcare, business, education, social science)
- **Format:** Instruction-following (user query → assistant critique)

### Training Configuration
```python
LoRA Config:
- Rank (r): 16
- Alpha: 32
- Target modules: All attention layers
- Trainable parameters: 20M (0.25% of 8B)

Training:
- Epochs: 3
- Batch size: 4 (effective)
- Learning rate: 2e-4 (cosine schedule)
- Quantization: 4-bit NF4
- Hardware: Google Colab T4 GPU (free)
```

### Reproduction
To reproduce training:
1. Run `1_data_generation.ipynb` to generate training data
2. Run `2_model_training.ipynb` to fine-tune the model
3. Training takes ~3-4 hours on a T4 GPU

---

## 🙏 Acknowledgments

- **Course:** COMS 4995 - Applied Machine Learning, Columbia University
- **Instructor:** Spencer Luo
- **Base Model:** Meta's Llama-3-8B-Instruct
- **Training Platform:** Google Colab (free T4 GPU)
- **Libraries:** HuggingFace Transformers, PEFT, Gradio
