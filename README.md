# Continuous-Time Generative Market Microstructure Modeling

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Quant Research](https://img.shields.io/badge/Domain-High--Frequency_Trading-success.svg)
![Status](https://img.shields.io/badge/Status-Active_Research-orange.svg)

This repository implements a comprehensive framework for modeling and generating high-frequency Limit Order Book (LOB) microstructures using continuous-time generative models. The project bridges **Rough Path Theory** and **Point Processes** to reconstruct complex market dynamics, effectively capturing both the continuous diffusion of asset prices and the data-driven discrete jumps inherent in micro-market environments.

To validate the practical utility of these generative models, the synthesized microstructure paths are directly integrated into a fully tensorized **Deep Hedging** environment. This allows for rigorous evaluation of how different microstructure assumptions impact non-linear slippage and hedging-loss in options pricing.

## Core Architecture & Methodology

The project decomposes high-frequency market dynamics into continuous evolution and discrete jumps, integrating them via a hybrid deep learning architecture:

1. **Structural Feature Representation (Path-Signature)**
   * Utilizes the `signatory` library to extract the truncated Signature of historical LOB paths as a macroscopic conditioning context. Compared to traditional RNNs/CNNs, the Path-Signature provides a parameter-free, highly efficient representation of cross-covariances and higher-order rough path features.
2. **Multivariate Discrete Jump Modeling (Neural-Hawkes)**
   * Replaces crude static jump thresholds with a data-driven **dynamic $k$-Sigma volatility test** to identify true market jumps.
   * Constructs a multivariate Neural-Hawkes process to independently fit the event arrival intensities and self-/mutually-exciting clustering effects across different LOB levels (price and volume).
3. **Continuous Diffusion & Hybrid Generation (Neural-SDE)**
   * Employs conditioned Neural-SDEs to model the continuous Brownian motion (drift and diffusion) of the underlying asset.
   * Implements an **SDE-Hawkes Hybrid** forward simulator: the model autoregressively fuses the sudden impact of Hawkes-predicted large orders with the continuous diffusion of the SDE. It incorporates autoregressive stabilizers (mean-reversion and probability throttling) to prevent long-sequence divergence.
4. *(Roadmap)* **Schrödinger-Bridge (SBM) Extension**
   * Pre-designed interfaces for dynamic Schrödinger Bridges, aiming to further push the theoretical upper bound of extreme thick-tail distribution fitting via Optimal Transport constraints.

## Multi-Dimensional Microstructure Evaluation

The system rejects single-loss-value evaluations, instead employing a rigorous, multi-dimensional discriminator based on quantitative stylized facts:

* **Path-MMD (Signature Space)**: Quantifies the overall distributional discrepancy between generated paths and real FI-2010 market paths in a non-linear feature space.
* **Jump Distribution Fitting (Wasserstein Distance)**: Uses 1D W1-Distance to evaluate the model's ability to reconstruct the thick-tailed characteristics of extreme jump magnitudes.
* **Intra-Sequence Clustering Behavior (ACF MSE)**: Fits the Mean Squared Error of the Autocorrelation Function (ACF) of absolute returns to verify if the Hawkes process successfully captures volatility clustering.
* **Order Flow Intensity Error**: Validates the reconstruction accuracy of abnormal order arrival frequencies per unit of time.

> **Dynamic Routing (Winner-Takes-All):** During the evaluation phase, the system automatically benchmarks the pure Neural-SDE (no-jump assumption) against the Hybrid model (jump assumption) using Path-MMD, dynamically routing the winning micro-dynamics to the downstream backtesting engine.

## Downstream Application: Deep Hedging & Impact Reconstruction

To demonstrate the empirical value of the generated paths, a Deep Hedging reinforcement learning environment is built purely in PyTorch, abandoning traditional low-frequency infinite-liquidity assumptions.

* **Microstructure Friction Modeling**: Replaces fixed transaction fees with a highly realistic non-linear impact cost function:  
  $\text{Cost} = (\text{Spread} / 2) \cdot |\Delta| + \lambda \cdot (\Delta^2 / \text{Volume})$.  
  This captures both the fixed cost of crossing the bid-ask spread and the non-linear slippage caused by depleting top-of-book liquidity.
* **AI Hedging Agent**: Trains a Multi-Layer Perceptron (MLP) as the hedger to output dynamic Delta positions on high-frequency paths, minimizing the variance of the Terminal PnL (including friction costs).
* **Baseline Comparison**: Under generated paths containing extreme jumps, the AI hedging strategy demonstrates a significant ability to truncate the left-tail risk inherent in static Black-Scholes Delta hedging.

### Performance Showcase

--------------------------------------------------
评估指标 (简单模型)        | 纯 Neural-SDE    | SDE-Hawkes 混合  
--------------------------------------------------
Path-MMD (Sig空间)     | 1136.7476       | 549.1089       
跳跃分布 W1 距离           | 0.0077          | 0.9998         
序列聚集 ACF MSE         | 0.4189          | 0.0000         
订单流强度误差              | 25.6395         | 97.9990        
--------------------------------------------------

 The Deep Hedger successfully mitigates extreme tail risks.* > *Right: Hedging trajectory under jump-diffusion paths. The AI strategy (blue) exhibits more resilient and forward-looking position management during sudden price jumps compared to the traditional BS Delta (red).*

## Repository Structure

```text
├── data_pipeline.py         # Rough path interpolation, Signature extraction & dynamic k-Sigma jump testing
├── generator_sde.py         # Continuous-time Neural-SDE (Drift & Diffusion networks)
├── generator_hawkes.py      # Multivariate Neural-Hawkes (Jump intensity & mark networks)
├── eval_path_mmd.py         # Kernel-based Path-MMD distribution discrepancy metric
├── eval_microstructure.py   # W1 distance, ACF clustering, and intensity error evaluators
├── deep_hedging_env.py      # Pure PyTorch non-linear impact cost & options hedging environment
└── main_evaluation.py       # End-to-end pipeline: Large-sample training, dynamic routing, and backtesting
