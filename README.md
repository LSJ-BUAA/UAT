# Unsupervised Adversarial Training (UAT)

## Introduction

This repository provides the simulation code of the following submitted paper.

> Shengjie Liu and Chenyang Yang, "Enhancing Out-of-Distribution Generalization in Learning Wireless Resource Allocation via Unsupervised Adversarial Training," submitted, 2026.

## Usage

- For Learning Hybrid Precoding
  - Generate Channel Datasets
    - Use `hybrid precoding\channels\matCode\mainGenChannel.m` to generate datasets with *QuaDRiGa*. These different channel distributions are modified from `foundation\quadriga_TR38901_single_cell_channel.m`, which includes detailed annotations.
    - Use `hybrid precoding\channels\pyCode\genDeepMIMO.py` and `genRealWorld.py` to generate datasets with *DeepMIMO* and *Ultra Dense Indoor MaMIMO CSI Dataset*.
  - Train and Test DNNs
    - Use `hybrid precoding\learning\train.py` to train DNNs with conventional unsupervised learning or UAT.
    - Use `hybrid precoding\learning\test.py` to test well-trained DNNs on various distributions.
  - Compare With Numerical Algorithms
    - Use `hybrid precoding\numerical algorithms\mainPrecode.m` to obtain the performance of numerical algorithms.  
- For Learning Cell-free Power Allocation
  - Generate Channel Datasets
    - Use `cell-free power allocation\channels\matCode\mainGenChannel.m` to generate datasets with *QuaDRiGa*.
    - Use `cell-free power allocation\channels\pyCode\genDeepMIMO.py` and `genRealWorld.py` to generate datasets with *DeepMIMO* and *Ultra Dense Indoor MaMIMO CSI Dataset*.
  - Train and Test DNNs
    - Use `cell-free power allocation\learning\train.py` to train DNNs with PDL or EUAT.
    - Use `cell-free power allocation\learning\test.py` to test well-trained DNNs on various distributions.
  - Compare With Numerical Algorithms
    - Use `cell-free power allocation\numerical algorithms\mainPowerAllocate.m` to obtain the performance of numerical algorithms.
