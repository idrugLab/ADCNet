# ADCNet
semi-supervised learning for ADC property prediction.
![image](https://github.com/idrugLab/ADCNet/blob/main/ADCNet.png)

# Requried package: 
## Example of ESM-2 environment installation：
```ruby
conda create -n esm-2 python==3.9
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
```

## Example of ADCNet environment installation：
```ruby
conda create -n ADCNet python==3.7
pip install tensorflow==2.3
pip install rdkit
pip install numpy
pip install pandas
conda install -c openbabel openbabel
pip install matplotlib
pip install hyperopt
pip install scikit-learn
pip install torch
pip install openpyxl
```

## Examples of obtaining embeddings for antibodies or antigens.
```ruby
conda activate esm-2
python ESM-2.py
```
After completion of the run, you will find a .pkl file in the current directory. It is a dictionary where the keys are ADC IDs (if there is no ADC ID, you can add a column with numerical values to the original data and name it ADC ID), and the values are tensors of 1280 dimensions.

## Examples of training ADCNet.
First, run ESM-2.py to obtain embeddings for the heavy chain, light chain, and antigen of the antibody. The code will save these embeddings into three pkl files.
Secondly, Ensure that each data entry contains the DAR value.
Finally, Create a folder named "medium3_weights" and place the file "bert_weightsMedium_20.h5" from this repository into that folder.
```ruby
conda activate ADCNet
python class.py
```
## Examples of using ADCNet to inference.
First, run ESM-2.py to obtain embeddings for the heavy chain, light chain, and antigen of the antibody. The code will save these embeddings into three pkl files.
Secondly, Ensure that each data entry contains the DAR value.
Finally, Create a folder named "classification_weights" and place the file "ADC_9.h5" from this repository into that folder.
```ruby
conda activate ADCNet
python inference.py
```
## Using ADCNet for predictions
```ruby
You can visit the (https://ADCNet.idruglab.cn) website to make predictions.
```


