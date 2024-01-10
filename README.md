# AdamL: A Fast Adaptive Gradient Method

This repository contains an implementation of the AdamL optimizer, a novel variant of the Adam optimizer that incorporates loss function information to achieve better generalization in deep learning models. The AdamL optimizer is detailed in the paper "AdamL: A fast adaptive gradient method incorporating loss function" ([arXiv:2312.15295](https://arxiv.org/abs/2312.15295)). This implementation is compatible with PyTorch and aims to provide a more efficient training process by dynamically adjusting the learning rate.


## Introduction

AdamL enhances the standard Adam optimizer by considering the loss function during optimization, which helps in achieving faster convergence and lower objective function values compared to Adam, EAdam, and AdaBelief. It has been shown to perform well across various deep learning tasks, including training convolutional neural networks (CNNs), generative adversarial networks (GANs), and long short-term memory (LSTM) networks. Notably, AdamL can linearly converge under certain conditions, without the need for manual learning rate adjustments in the later stages of training CNNs.

## Benefits

* Faster Convergence: AdamL typically achieves faster convergence compared to other optimizers.
* No Manual Learning Rate Tuning: Unlike other variants, AdamL does not require manual learning rate adjustments, turn your scheduler off!

## Installation

To integrate the AdamL optimizer into your PyTorch projects, use the following pip command:

pip install git+https://github.com/andrewjc/pytorch-adaml.git

## Usage

    import torch
    from adaml import AdamL

    model = UberComplexModel()
    optimizer = AdamL(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    for epoch in range(num_epochs):
        for batch in dataloader:
            # Your training loop
            optimizer.zero_grad()
            output = model(input)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

## Documentation

Note: I am unsure how AdamL works alongside learning rate schedulers yet. In my testing, i'm turning my scheduler off and i suggest doing the same for now.

The `AdamL` class takes the following parameters:

- `params` (iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
- `lr` (float, optional): Learning rate (default: 1e-3).
- `betas` (Tuple[float, float], optional): Coefficients for gradient and square gradient moving averages (default: (0.9, 0.999)).
- `eps` (float, optional): Term for numerical stability in the denominator (default: 1e-8).
- `weight_decay` (float, optional): Weight decay coefficient (default: 0).

@article{2023AdamL,
title={AdamL: A fast adaptive gradient method incorporating loss function},
author={Lu Xia, Stefano Massei},
journal={arXiv preprint arXiv:2312.15295},
year={2023}
}