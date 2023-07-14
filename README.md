# EvoAug-TF Analysis

This repository performs the analysis from "   " by Yiyang Yu and Peter K Koo. Code in this repository is shared under the MIT License. For additional information, see the [EvoAug-TF repository](https://github.com/54yyyu/evoaug-tf.git) and EvoAug-TF documentation on ...


#### Install:
```
pip install git+https://github.com/54yyyu/evoaug-tf_analysis.git
```

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

- Example analysis: https://colab.research.google.com/drive/1sCYAL133F1PPbn7aGOxeQTFW-6fpLo4r?authuser=1#scrollTo=bcXlZ57uncra
- Example Ray Tune with Population Based Training: https://colab.research.google.com/drive/1NG8DrELTdmZPOw0RmaeNky0DZ5m2jpXY?authuser=1#scrollTo=NqfTP34ZsqbE
- Example Ray Tune with Asynchronous Hyperband Algorithm: https://colab.research.google.com/drive/1mzKeXKSfkEfe9o-P-MhqQokLoW7Dv-Jk?authuser=1#scrollTo=qofIghsSs7Kf
