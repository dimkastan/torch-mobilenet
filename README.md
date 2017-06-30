# torch-mobilenet
MobileNet implementation for torch based on Pytorch Equivalent: https://github.com/marvis/pytorch-mobilenet 


## Description
This is a Torch implementation of Mobilenet architecture.


## TODO
a. Add bias once the nn library solves the issue with Bias on DepthWiseSeparable convolution.
<br />
b. Add training on imagenet and food101.
<br />
c. Add pretrained model and demo
<br />

## Important Note
Training is very slow. In particular, it takes about 300msec to process a batch of 10 images (30msec/image) on a NVIDIA GTX-1070.



Please feel free to contact me for any comments or suggestions.
