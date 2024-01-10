# AdamL: A Fast Adaptive Gradient Method

This repository hosts the implementation of the AdamL optimizer, as described in the paper "AdamL: A fast adaptive gradient method incorporating loss function" ([arXiv:2312.15295](https://arxiv.org/abs/2312.15295)). AdamL is an advanced optimization algorithm for deep learning, enhancing the standard Adam optimizer by incorporating adaptive learning rate adjustments based on the loss function. This implementation is designed for integration with PyTorch.

## Introduction

AdamL is a variant of the Adam optimizer that dynamically adjusts the learning rate based on the adaptive and non-adaptive mode considerations. This approach results in more efficient training of neural networks, particularly in complex tasks. 

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