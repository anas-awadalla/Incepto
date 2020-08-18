# Details

This bias detector is meant to help ML engineers find and visualize underlying biases in their models. This detector currently only supports Image and Text Classification. Stay tuned for more model types!
Follow the instructions in the sidebar to get started.

## Parameters

**Models to Use**  
* `Face Classification`: Subset the 1kG data to the 55 SNPs listed in the manuscript.
* `Text Classification`: Subset the 1kG data to the 128 SNPs listed in the manuscript.

**Tokenizer**
* 'T5': Choose a T5 tokenizer.
* 'BERT': Choose a BERT tokenizer.

**Max Text Length**
* 'Max Text Length': Choose a text length that fits your model.

**Image Data Params**
* `Resize`: Resize Image to a certain dimension.
* `Center Crop`: Center Crop Image to a certain dimension.
* `Normalize`: Normalize image by [[0.5,0.5,0.5],[0.5,0.5,0.5]].

**Class Params**
* 'Social Class': Set social class order such as 2,1,0,3.

## Code  
The code used to process this data is available on [GitHub](https://github.com/ajfrai/Pytorch-BiasDetection).
