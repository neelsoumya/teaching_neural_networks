# Convolutional Neural Networks for Biologists

A practical introduction to CNNs designed specifically for researchers in the biological sciences.

## Course Overview

This course introduces convolutional neural networks (CNNs) with biological applications in mind. No deep computer science background required—just curiosity and basic Python knowledge.

## Prerequisites

- Basic Python programming
- Familiarity with NumPy arrays
- Understanding of basic biology/microscopy concepts
- High school level mathematics

## Course Structure

1. **Module 1: Introduction to CNNs** (`01_introduction.md`)
   - What are CNNs and why biologists should care
   - Real-world biological applications
   - Basic neural network concepts

2. **Module 2: How CNNs Work** (`02_how_cnns_work.md`)
   - Convolution operations explained visually
   - Pooling and activation functions
   - Architecture basics

3. **Module 3: Hands-On with CNNs** (`03_hands_on_tutorial.md`)
   - Building your first CNN
   - Training on biological image data
   - Evaluating model performance

4. **Module 4: Biological Applications** (`04_biological_applications.md`)
   - Cell classification
   - Microscopy image analysis
   - Medical imaging
   - DNA/protein sequence analysis

## Setup Instructions

### 1. Install Python
Ensure you have Python 3.8 or higher installed.

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv cnn_course_env
source cnn_course_env/bin/activate  # On Windows: cnn_course_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python verify_installation.py
```

## Code Examples

All code examples are in the `code/` directory:
- `example_01_simple_cnn.py` - Basic CNN architecture
- `example_02_cell_classifier.py` - Cell image classification
- `example_03_transfer_learning.py` - Using pre-trained models
- `utils.py` - Helper functions for visualization

## Running Examples

```bash
cd code
python example_01_simple_cnn.py
```

## Dataset

The course uses sample microscopy images. Real datasets can be obtained from:
- Cell Image Library (cellimagelibrary.org)
- Broad Bioimage Benchmark Collection
- Your own microscopy data

## Learning Path

1. Read modules in order (01 → 04)
2. Run code examples after each module
3. Experiment with your own biological images
4. Complete the exercises in each module

## Getting Help

- Check the FAQ section in each module
- Review code comments carefully
- Consult documentation: [PyTorch](https://pytorch.org/docs/) | [scikit-image](https://scikit-image.org/)

## Additional Resources

- Book: "Deep Learning for the Life Sciences" (Ramsundar et al.)
- Course: fast.ai's Practical Deep Learning
- Paper: "Deep learning in microscopy" (Belthangady & Royer, 2019)

## License

Educational use only. Code examples provided as-is for learning purposes.

---

**Happy Learning!** Remember: CNNs are just tools. Your biological insight is what makes them powerful.
