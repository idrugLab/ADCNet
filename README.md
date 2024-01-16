# ADCNet
semi-supervised learning for ADC property prediction.
![image](https://github.com/idrugLab/ADCNet/blob/main/ADCNet.png)

# Requried package: 
## Example of ESM-2 environment installation：
--conda create -n esm-2 python==3.9

--pip install fair-esm  # latest release, OR:
--pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch

## Example of ADCNet environment installation：
--conda create -n ADCNet python==3.7

--pip install tensorflow==2.3

--pip install rdkit

--pip install numpy

--pip install pandas

--conda install -c openbabel openbabel

--pip install matplotlib

--pip install hyperopt

--pip install scikit-learn

--pip install torch
## Examples of obtaining embeddings for antibodies or antigens.
--conda activate esm-2
--python ESM-2.py
