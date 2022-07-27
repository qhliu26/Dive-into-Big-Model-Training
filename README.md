# Dive into Big Model Training

📰 Report Link [[here]](https://arxiv.org/abs/2207.11912)

📫 Contact me qhliu26@gmail.com

The increasing scale of model size and continuous improvement of performance herald the arrival of the Big Model era. In this report, we explore what and how the big model training works by diving into training objectives and training methodologies. Specifically,training objectives describe how to leverage web-scale data to develop extremely capable and incredibly large models based on self-supervised learning, and training methodologies which are based on distributed training describe how to make big model training a reality. We summarize the existing training methodologies into three main categories: training parallelism, memory-saving technologies, and model sparsity design. Training parallelism can be categorized into data, pipeline, and tensor parallelism according to the dimension of parallelism that takes place. Memory-saving technologies are orthogonal and complementary to training parallelism. And model sparsity design further scales up the model size with a constant computational cost.

## Uesful Repositories

+ PyTorch: https://github.com/pytorch/pytorch
+ TensorFlow: https://github.com/tensorflow/tensorflow
+ Mesh TensorFlow: https://github.com/tensorflow/mesh
+ Megatron-LM: https://github.com/NVIDIA/Megatron-LM
+ DeepSpeed: https://github.com/microsoft/DeepSpeed
+ Fairscale: https://github.com/facebookresearch/fairscale
+ Colossal-AI: https://github.com/hpcaitech/ColossalAI
+ OneFlow: https://github.com/Oneflow-Inc/oneflow

## BM Background

| Year | Title                                                        | Intro                                                        |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2017 | [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409) | empirical characterization of generalization error and model size growth as training sets grow |
| 2020 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Performance depends strongly on scale, weakly on model shape |
| 2021 | [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) | A foundation model is any model that is trained on broad data at scale and can be adapted to a wide range of downstream tasks |
| 2022 | [The 2022 AI Index](http://export.arxiv.org/abs/2205.03468)  | Language models are more capable than ever, but also more biased |

## Training Parallelism

### Data Parallelism

| Year | Title                                                        | Intro                                             |
| ---- | ------------------------------------------------------------ | ------------------------------------------------- |
| 2009 | [Bandwidth optimal all-reduce algorithms for clusters of workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) | All-Reduce Architecture                           |
| 2011 | [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730#:~:text=A%20Lock-Free%20Approach%20to%20Parallelizing%20Stochastic%20Gradient%20Descent,tasks.%20Several%20researchers%20have%20recently%20proposed%20schemes%20to) | Asynchronous SGD                                  |
| 2014 | [Scaling Distributed Machine Learning with the Parameter Server](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf) | Traditional Centralized Architecture              |
| 2016 | [GeePS: Scalable deep learning on distributed GPUs with a GPU-specialized parameter server](https://www.pdl.cmu.edu/PDL-FTP/CloudComputing/GeePS-cui-eurosys16.pdf) | offload temporarily unused parameters back to CPU |
| 2020 | [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704) | PyTorch DDP Implementation                        |
| 2020 | [A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters](https://www.usenix.org/system/files/osdi20-jiang.pdf) | BytePS                                            |

### Tensor Parallelism

| Year | Title                                                        | Intro                                                        |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2019 | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | 1D tensor parallelism for transformer MLP and self-attention |
| 2021 | [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343) | 2D TP based on SUMMA                                         |
| 2021 | [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500) | 2.5D TP                                                      |
| 2021 | [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450) | 3D TP                                                        |

### Pipeline Parallelism

| Year | Title                                                        | Intro                              |
| ---- | ------------------------------------------------------------ | ---------------------------------- |
| 2018 | [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) | Pipeline Parallelism from Google   |
| 2019 | [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf) | 1F1B microbatch scheduling         |
| 2020 | [Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2006.09503) | PipeDream-flush and PipeDream-2BW  |
| 2021 | [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) | Interleaved 1F1B pipeline schedule |

## Mixture-of-Expert

| Year | Title                                                        | Intro                                                        |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2017 | [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | ensembling implemented with a gating mechanism connecting multiple experts |
| 2020 | [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) | replaces transformer FFN with MoE layer                      |
| 2021 | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | scales the model size up to trillions of parameters          |
| 2021 | [Go Wider Instead of Deeper](https://arxiv.org/abs/2107.11817) | WideNet uses individual LN to transform semantic representations |
| 2022 | [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368) | let experts select the top-k tokens                          |

## Memory Saving Design

### ZeRO

| Year | Title                                                        | Intro                                                        |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2019 | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | Zero Redundancy Optimizer                                    |
| 2021 | [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840) | offloading data and compute to CPU                           |
| 2021 | [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) | heterogeneous system technology leverages GPU, CPU, and NVMe memory to allow for unprecedented model scale |

### Mix Precision Training

| Year | Title                                                        | Intro                                            |
| ---- | ------------------------------------------------------------ | ------------------------------------------------ |
| 2017 | [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | Speed  up training and save memory               |
| 2018 | [Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/abs/1807.11205) | training AlexNet with 95 epochs within 4 minutes |
