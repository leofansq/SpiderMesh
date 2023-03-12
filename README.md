# SpiderMesh: Spatial-aware Demand-guided Recursive Meshing for RGB-T Semantic Segmentation

![architecture](./img/SpiderMesh.png)

> For technical details, please refer to:
>
> [xxx](xxx)

### (0) Abstract

For semantic segmentation in urban scene understanding, RGB cameras alone often fail to capture a clear holistic topology, especially in challenging lighting conditions. Thermal signal is an informative additional channel that can bring to light the contour and fine-grained texture of blurred regions in low-quality RGB image. Aiming at RGB-T (thermal) segmentation, existing methods either use simple passive channel/spatial-wise fusion for cross-modal interaction, or rely on heavy labeling of ambiguous boundaries for fine-grained supervision. We propose a Spatial-aware Demand-guided Recursive Meshing (SpiderMesh) framework that: 1) proactively compensates inadequate contextual semantics in optically-impaired regions via a demand-guided target masking algorithm; 2) refines multimodal semantic features with recursive meshing to improve pixel-level semantic analysis performance. We further introduce an asymmetric data augmentation technique M-CutOut, and enable semi-supervised learning to fully utilize RGB-T labels only sparsely available in practical use. Extensive experiments on MFNet and PST900 datasets demonstrate that SpiderMesh achieves new state-of-the-art performance on standard RGB-T segmentation benchmarks.

### (1) Code

**Code and pretrained models will be released upon publication.**

### (2) Performance

* Supervised learning 

![performance](./img/performance.png)

* Semi-supervised learning
    
![ssl_performance](./img/ssl_performance.png)

* Complexity

![complexity](./img/complexity.png)






