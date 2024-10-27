# 1D Tokenizer

Simple 1D image tokenizer from the paper [_An Image is Worth 32 Tokens for Reconstruction and Generation_](https://arxiv.org/pdf/2406.07550)

![Image Tokenizer](./assets/encoder.png)

# Image Tokenizer

A neural architecture for encoding images into sequences of discrete tokens, enabling efficient image compression and representation learning through the use of Vision Transformers and Vector Quantization.

## Overview

The Image Tokenizer converts images into sequences of discrete tokens using a two-stage process:

1. Vision Transformer (ViT) for learning spatial relationships
2. Vector Quantization (VQ) for discretization

This architecture enables efficient image compression, learned discrete representations, and interpretable latent spaces suitable for downstream tasks.

## Architecture Details

### Image Tokenizer Pipeline

The image tokenizer processes input images $x \in \mathbb{R}^{H \times W \times C}$ through the following stages:

1. Patch embedding and tokenization
2. Transformer-based contextual encoding
3. Vector quantization
4. Discrete token generation

### Vision Transformer (ViT)

The Vision Transformer processes images through several key stages:

1. **Patch Embedding:**

   - Input image $x \in \mathbb{R}^{H \times W \times C}$ is divided into $N = \frac{HW}{P^2}$ patches
   - Each patch $x_p \in \mathbb{R}^{P^2 \cdot C}$ is projected to dimension $D$
   - Result: sequence of patch embeddings $z_0 \in \mathbb{R}^{N \times D}$

2. **Position Encoding:**

   - Learned position embeddings $E_{pos} \in \mathbb{R}^{N \times D}$ added to patch embeddings
   - Input sequence: $z_0 + E_{pos}$

3. **Transformer Encoding:**
   - $L$ layers of multi-head self-attention and MLP blocks
   - Layer $l$ computation:
     $$\begin{aligned}z'_l=\text{MSA}(\text{LN}(z_{l-1}))+z_{l-1}z_l=\text{MLP}(\text{LN}(z'_l))+z'_l\end{aligned}$$
   - Output: contextual representations $z_L \in \mathbb{R}^{N \times D}$

### Vector Quantization (VQ)

The VQ layer maps continuous latent vectors to discrete tokens:

1. **Codebook:**

   - Contains $K$ embedding vectors: $\{e_k\}_{k=1}^K$ where $e_k \in \mathbb{R}^D$
   - Learned during training through straight-through gradient estimation

2. **Quantization Process:**

   - For each input vector $z_i$, find nearest codebook vector:
     $$k(i) = \arg\min_k \|z_i - e_k\|_2$$
   - Replace with selected codebook vector:
     $$z_q^i = e_{k(i)}$$

3. **Training Objectives:**

   - Codebook loss: $\|sg(z) - e\|_2^2$
   - Commitment loss: $\beta\|z - sg(e)\|_2^2$
   - Where $sg()$ is the stop-gradient operator

4. **Token Generation:**
   - Each quantized vector replaced by codebook index
   - Final output: sequence of $N$ discrete tokens $\{k(i)\}_{i=1}^N$

## Mathematical Framework

### Image Processing

For an input image with dimensions $H \times W$:

- Patch size $P$ results in $N = \frac{HW}{P^2}$ patches
- Each patch produces one embedding in final sequence
- Example: 256×256 image with 64×64 patches yields 16 embeddings

### Attention Mechanism

Multi-head attention computed as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:

- $Q, K, V \in \mathbb{R}^{N \times D}$ are query, key, value matrices
- $d_k$ is scaling factor equal to head dimension

### Vector Quantization

The quantization operation $q(z)$ is defined as:
$$q(z) = e_k \text{ where } k = \arg\min_j \|z - e_j\|_2$$

Total loss:
$$\mathcal{L} = \mathcal{L}_\text{reconstruction} + \|sg(z) - e\|_2^2 + \beta\|z - sg(e)\|_2^2$$

## Model Configuration

Typical hyperparameters:

- Image size: 256×256
- Patch size: 64×64
- Model dimension: 1024
- Number of heads: 16
- Number of layers: 12
- Codebook size: 8192
- $\beta$ (commitment cost): 0.25

## Input-Output Specifications

Input:

- RGB images: $\mathbb{R}^{H \times W \times 3}$
- Normalized to [-1, 1] range

Output:

- Sequence of discrete tokens: $\{0, ..., K-1\}^N$
- Token sequence length = $\frac{HW}{P^2}$

## Performance Characteristics

1. **Compression Rate:**

   - Input: $H \times W \times 3$ bytes
   - Output: $\frac{HW}{P^2} \times \log_2(K)$ bits
   - Example compression ratio ≈ 24:1

2. **Computational Complexity:**

   - Attention: $O(N^2D)$ per layer
   - Vector Quantization: $O(NKD)$

3. **Memory Usage:**
   - Codebook: $O(KD)$ parameters
   - Transformer: $O(L D^2)$ parameters
