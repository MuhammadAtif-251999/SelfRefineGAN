# Enhancing Low-Light Image Quality with Self-RefineGAN: A Lightweight GAN Approach

Official PyTorch implementation of **Self-RefineGAN**, a lightweight Generative Adversarial Network designed for high-fidelity low-light image enhancement on resource-constrained edge devices.

## 🚀 Overview
Self-RefineGAN addresses the trade-off between enhancement quality and computational efficiency. Our framework introduces a training paradigm where a **Refinement Network (RFN)** guides an **Extreme Lightweight Residual Gated Network (XLRGN)**.

* **Training Phase:** XLRGN and RFN are trained iteratively using a combination of Adversarial, Perceptual, and Texture Energy Loss.
* **Inference Phase:** The RFN is decoupled, leaving only the ultra-lightweight XLRGN (0.025M parameters) for real-time deployment.

## ✨ Key Contributions
* **XLRGN Generator:** A gated residual network optimized for edge devices.
* **Refinement Model (RFN):** A curriculum-based teacher network that improves training stability.
* **Efficiency:** Achieves state-of-the-art results with only **3.69 GFLOPs**.

## 📊 Datasets
The following datasets were used for training and evaluating **Self-RefineGAN**:

* **[LOLv1 Dataset](https://huggingface.co/datasets/geekyrakshit/LoL-Dataset)**: 500 paired low-light/normal-light images.
* **[LOLv2 Dataset](https://huggingface.co/datasets/okhater/lolv2-synthetic)**: Real and Synthetic subsets.
* **[MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/)**: 5,000 professionally retouched photos.
* **[SICE Dataset](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view)**: Multi-exposure image sequences.

## 🚀 Status
Official implementation of the Self-RefineGAN. Code and pre-trained models will be released upon paper publication.

## 📬 Contact
If you are a reviewer or researcher with urgent questions, 
please contact the author at [matifblogger@gmail.com].

Muhammad Atif Department of Computer Science and Engineering

Southeast University, Nanjing, China
