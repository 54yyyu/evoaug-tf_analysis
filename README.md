# EvoAug-TF Analysis

This repository performs the analysis from "EvoAug-TF: Extending evolution-inspired data augmentations for genomic deep learning to TensorFlow" by Yiyang Yu, Shivani Muthukumar and Peter Koo. Code in this repository is shared under the MIT License. For additional information, see the [EvoAug-TF repository](https://github.com/p-koo/evoaug-tf.git) and EvoAug-TF documentation on [ReadtheDocs](https://evoaug-tf.readthedocs.io)


#### Dependencies:
```
tensorflow 2.11.0+cu114
evoaug-tf
scipy 1.7.3
h5py 3.1.0
matplotlib 3.2.2
numpy 1.21.6
```

Note: For older versions of tensorflow, the Adam optimizer will need to be modified accordingly as the arguments has changed from version 2.8


#### Example on Google Colab:


- Example DeepSTARR analysis: https://colab.research.google.com/drive/11TA02v-azuqAIV5s3sCbWTH-C5W7_KMA?usp=sharing
- Example ChIP-seq analysis: https://colab.research.google.com/drive/1fzpH2Qv8RFNzMvIDRJUGnUTaacFMIJBV?usp=sharing
