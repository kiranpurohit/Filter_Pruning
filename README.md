##  Accurate and Efficient Channel pruning via Orthogonal Matching Pursuit
Pytorch Implementation of the following paper: 

"Accurate and Efficient Channel pruning via Orthogonal Matching Pursuit", AIMLSystems 2022.
Kiran Purohit, Anurag Parvathgari, Soumi Das and Sourangshu Bhattacharya


### Overview
---
The deeper and wider architectures of recent convolutional neural networks (CNN) are responsible for superior performance in computer vision tasks. However, they also come with an enormous model size and heavy computational cost. Filter pruning (FP) is one of the methods applied to CNNs for compression and acceleration. Various techniques have been  recently proposed for filter pruning. We address the limitation of the existing state-of-the-art method and motivate our setup. We develop a novel method for filter selection using sparse approximation of filter weights. We propose an orthogonal matching pursuit (OMP) based algorithm for filter pruning (called FP-OMP). We also propose FP-OMP Search, which address the problem of removal of uniform number of filters from all the layers of a network. FP-OMP Search performs a search over all the layers with a given batch size of filter removal. We evaluate both FP-OMP and FP-OMP Search on benchmark datasets using standard ResNet architectures. Experimental results indicate that FP-OMP Search consistently outperforms the baseline method (LRF) by nearly 0.5 - 3 %. We demonstrate both empirically and visually, that FP-OMP Search prunes different number of filters from different layers. Further, timing profile experiments show that FP-OMP improves over the running time of LRF.

#### Sample command
Pruning ResNet-32 model using our OMP method on CIFAR-100 dataset.
```
python omp.py
