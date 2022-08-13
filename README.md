# Readme


## Description

This package includes the python code of the OLI2DS(Online learning from Incomplete and Imbalanced Data Streams) which focus on the problem of learning from data streams with incomplete feature space and class-imbalance.

## Normalization Codes

```python
# dataset is the original data that read from data set file.
dataset = np.array(dataset).astype(np.float) # transform the type of original data to 'numpy.float' type
dataset[:, :-1] = preprocessing.scale(dataset[:, :-1]) # normalized data
```

## Requirement

This package was developed with Python 3.6.5. The environment can be find in `requirement.yaml` file.
 
 ## Notes
 
 - This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Dian-Long You (youdianlong@sina.com).

- This package was developed by Mr. Jia-Wei Xiao (xiaojiaweix@163.com). For any problem concerning the code, please feel free to contact Mr. Xiao.
