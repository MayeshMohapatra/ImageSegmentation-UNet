# ImageSegmentation-UNet
Deep learning model based on the u-net architecture for image segmentation on human images dataset.
# Model
The model uses the u-net architecture proposed by *Ronneberger et al* in the paper titled *U-net: Convolutional Networks for for Biomedical Image Segmentation* , the paper can be accessed through [this arxiv link](https://arxiv.org/abs/1505.04597) . 
## Network Architecture
As described in the paper, the network architecture consists of a contracting path and a expansive path. <br>
The contracting path consisting of repeated application of two 3x3 unpadded convolution layers each consisting of a 2x2 max pooling with stride 2 after the convolution layer. After each downsampling, we double the number of feature channels. <br>
The expansive path consists of an upsampling of the feature map which is followed by a 2x2 up-convolution to half the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path at each level, and two 3x3 convolutions, each convolution layer followed by a ReLU. <br>
The upsampling is followed by a final 1x1 convolution layer to map the 64-component vector to the desired number of classes. Totally, 23 convolution layers are used in the architecture. 
![image](https://user-images.githubusercontent.com/53568572/132100326-9622f3f9-4a45-4b52-8ee4-ff53ff157f6a.png)
<b>Image Source:</b> U-Net: Convolutional Networks for Biomedical Image Segmentation <br><br>
For a seamless tiling of the segmentation map(output) the authors make it a point that the input tile size is such that the 2x2 max-pooling operations are applied to a layer with an even x- and y- size.

# Dataset
The dataset used was from kaggle and can be freely downloaded and used from this link: [Person Segmentation](https://www.kaggle.com/dataset/b9d4e32be2f57c2901fc9c5cd5f6633be7075f4b32d73348a6d5db245f2c1934) 
The dataset comprises of images and segmentation masks corresponding to the images.
The data comes from [supervisely](https://supervise.ly/)
More information about the dataset could be found in this [blog](https://hackernoon.com/releasing-supervisely-person-dataset-for-teaching-machines-to-segment-humans-1f1fc1f28469)
