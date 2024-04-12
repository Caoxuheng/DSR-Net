![TITLE](https://github.com/Caoxuheng/imgs/blob/main/OL.png)
# [Unsupervised Spectral Reconstruction from RGB images under Two Lighting Conditions](https://doi.org/10.1364/OL.517007)
This is an unsupervised unrolled deep network that recoconstructs the hyperspectral image from RGB images captures under two lighting conditions.  
# Abstract
Unsupervised spectral reconstruction (SR) aims to recover the hyperspectral image (HSI) from corresponding RGB images without annotations. Existing SR methods achieve it from a single RGB image, hindered by the significant spectral distortion. Although several deep learning-based methods increase the SR accuracy by adding RGB images, their networks are always designed for other image recovery tasks, leaving huge room for improvement. To overcome this problem, we propose a novel, to our knowledge, approach that reconstructs the HSI from a pair of RGB images captured under two illuminations, significantly improving reconstruction accuracy. Specifically, an SR iterative model based on two illuminations is constructed at first. By unfolding the proximal gradient algorithm solving this SR model, an interpretable unsupervised deep network is proposed. All the modules in the proposed network have precise physical meanings, which enable our network to have superior performance and good generalization capability. Experimental results on two public datasets and our real-world images show the proposed method significantly improves both visually and quantitatively as compared with state-of-the-art methods.  
# Requirements
## Environment
`Python3.8`  
`torch 1.12`,`torchvision 0.13.0`  
`Numpy`,`Scipy`  
*Also, we will create a Paddle version that implements FeafusFormer in AI Studio online for free!*
## Datasets
[`CAVE dataset`](https://www1.cs.columbia.edu/CAVE/databases/multispectral/), 
 [`Preprocessed CAVE dataset`](https://aistudio.baidu.com/aistudio/datasetdetail/147509).
# Note
For any questions, feel free to email me at caoxuhengcn@gmail.com.  
If you find our work useful in your research, please cite our paper ^.^
