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

**Team:** [Your Names Here]

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

## 📖 Usage Examples

### Example 1: Truncated Y-Axis

**Input Code:**
```python
import matplotlib.pyplot as plt

quarters = ['Q1', 'Q2', 'Q3', 'Q4']
profits = [98, 99, 100, 101]

plt.bar(quarters, profits)
plt.ylim(97, 102)  # Truncated!
plt.title('MASSIVE Profit Growth!')
plt.show()
```

**Model Output:**
```
Main Error: Truncated y-axis manipulation

The y-axis starts at 97 instead of 0, making a 3% increase 
appear as massive growth. This visual distortion exaggerates 
small differences and misleads viewers.

Fix: Start y-axis at 0 or use an appropriate scale that 
accurately represents the magnitude of change.
```

### Example 2: Multiple Testing

**Input Code:**
```python
from scipy.stats import ttest_ind
import numpy as np

control = np.random.normal(100, 15, 50)

for i in range(20):  # Testing 20 drugs
    treatment = np.random.normal(100, 15, 50)
    t, p = ttest_ind(control, treatment)
    if p < 0.05:
        print(f"Drug {i} works! p={p:.4f}")
```

**Model Output:**
```
Main Error: Multiple testing without correction

Running 20 tests at p<0.05 threshold expects ~1 false positive 
by chance. Without Bonferroni correction or family-wise error 
rate adjustment, the Type I error rate is inflated.

Fix: Apply Bonferroni correction (α = 0.05/20 = 0.0025) or use 
Holm-Bonferroni method.
```

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

## 📊 Evaluation

**Test Set:** 8 held-out error types

**Results:**

| Error Type | Detected? | Quality |
|------------|-----------|---------|
| Truncated Axis | ✅ | Excellent |
| Multiple Testing | ✅ | Excellent |
| Correlation→Causation | ✅ | Excellent |
| Simpson's Paradox | ✅ | Good |
| P-hacking | ✅ | Excellent |
| Missing Uncertainty | ✅ | Good |
| Omitted Confounder | ✅ | Excellent |
| Overplotting | ✅ | Good |

**Overall Accuracy:** 100% (8/8)

See `data/evaluation_results.csv` for detailed results.

---

## 🔗 Model Weights

Due to GitHub file size limits, trained model weights are hosted separately:

**Download Link:** [Google Drive Link] *(Add your link here)*

**Size:** ~20MB (LoRA adapters only)

**To use downloaded weights:**
1. Place `final_model/` folder in project root
2. Update model path in `3_demo_evaluation.ipynb` cell 2

---

## 🎥 Demo Video

**[Link to 10-minute presentation video]** *(Add your link here)*

The video demonstrates:
- Project motivation and novelty
- Technical approach and training pipeline
- Live demo with 3 examples
- Evaluation results and discussion

---

## 📝 Project Report

**[Link to full 8-page report]** *(Add your link here)*

The report includes:
- Detailed methodology
- Training data generation process
- Model architecture and fine-tuning approach
- Comprehensive evaluation results
- Limitations and future work

---

## ⚠️ Limitations

**Current Limitations:**
- Works best when code includes explicit error indicators
- Python-only (no support for R, Julia, SQL)
- Occasional formatting artifacts in outputs
- Limited to 15 pre-defined error types

**Future Improvements:**
- Detect errors from code structure alone (less reliance on comments)
- Multi-language support (R, Julia, SQL)
- Integration with Jupyter notebooks for real-time feedback
- Expand error taxonomy (causal inference, Bayesian reasoning)
- Larger training dataset (1000+ examples)

---

## 🤝 Contributing

This is an academic project, but suggestions are welcome!

**To contribute:**
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

**Areas for contribution:**
- Additional error types
- Multi-language support
- Improved error detection accuracy
- Better visualization generation

---

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

**Note:** The Llama-3 base model requires acceptance of Meta's license agreement.

---

## 🙏 Acknowledgments

- **Course:** COMS 4995 - Applied Machine Learning, Columbia University
- **Instructor:** [Instructor Name]
- **Base Model:** Meta's Llama-3-8B-Instruct
- **Training Platform:** Google Colab (free T4 GPU)
- **Libraries:** HuggingFace Transformers, PEFT, Gradio

---

## 📧 Contact

**Team Members:**
- [Name 1] - [email@columbia.edu]
- [Name 2] - [email@columbia.edu]
- [Name 3] - [email@columbia.edu] *(if applicable)*

**Project Page:** [Link to project website if any]

---

## 📚 Citation

If you use this work, please cite:
```bibtex
@software{dataviz_critic_2024,
  title={Data Visualization Critic: AI-Powered Statistical Code Review},
  author={[Your Names]},
  year={2024},
  institution={Columbia University},
  course={COMS 4995 - Applied Machine Learning}
}
```

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

*Last Updated: December 2024*
```

---

## Step 3: Create requirements.txt

Create `requirements.txt`:
```
transformers>=4.36.0
torch>=2.0.0
peft>=0.7.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
datasets>=2.16.0
trl>=0.7.0
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
pillow>=10.0.0
scikit-learn>=1.3.0
```

---

## Step 4: Create .gitignore

Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Data files (too large for GitHub)
*.jsonl
data/training_data_combined.jsonl
data/training_data_v*.jsonl

# Model weights (too large for GitHub)
lora_model*/
final_model/
*.bin
*.safetensors
adapter_*.bin

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary files
tmp/
temp/
*.tmp