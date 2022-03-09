# **Source Codes for Meta-Spec**

## Contents

- [Introduction](#introduction)
- [Package requirement](#package-requirement)
- [Pre-computing](#pre-computing)
- [Experiment dataset](#experiment-dataset)
- [Run Example](#run-example)
- [Input and output format](#input-and-output-format)
- [Instructions for Meta-Spec](#instructions-for-meta-spec)
- [Supplementary](#supplementary)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

Meta-Spec is a microbiome multi-label disease classification model based on explainable deep learning which is capable to detect multiple diseases simultaneously by integrating genotype data (microbiome features) and phenotype data (host variables).

## Package requirement

- torch==1.9.0+cu111
- deepctr-torch==0.2.7
- shap==0.35.0

```
sh init.sh
```

## Pre-computing


## Experiment dataset
Here we provide the dataset used in our paper. The dataset includes 11,936 subjects including 4437 healthy controls and 7502 patients with 844 ASVs and 56 host variables which are collected from *McDonald, D., et al.*.

\* McDonald, D., et al., American gut: an open platform for citizen science microbiome research. Msystems, 2018. 3(3): p. e00031-18.

## Run example

```
cd Codes/inference/
```

- Run host_variable_features.ipynb to generate host_variable.csv in data file
- Run rf.ipynb and lgb.ipynb to obtain results of Random Forest, Lightgbm
- Run meta_spec_without_hostvar.ipynb to get results of Meta-Spec without host variables
- Run meta_spec.ipynb to get results of Meta-Spec

## Input  format
A single sample is a patient with ASVs and host variables where the ASVs must be named begin with 'asv'.
| **SampleID** | **asv1**  /**age**|
| ----------------- | --------- |
| 1             | 0.1        |1      |
| 2             | 0.05      |3      |
| 3             | 0.07      |2      |

## Instructions for Meta-Spec
Here are basic steps to train Meta-Spec. First, use Dataset function to preprocess data where the pickle files are the indexes of training set and test set. Then, user df.get_train_test(0) to split the data and use df.deal_feat() to get features.
Besides, df.get_input(train, test) is able to obtain input for the Meta-Spec. Finally, use MMOE function to train the model and evaluation function to get AUC and f1-scores.

```
import torch
from model.mmoe_ifm_ew import MMOE
from utility.dataset import Dataset
from utility.eval_mmoe_train import evaluation

df = Dataset(data, 'train_index.pickle', 'test_index.pickle',  ['ibs', 'thyroid', 'migraine'])
train, test = df.get_train_test(0)
feats = df.deal_feat()
dnn_feature_columns, train_model_input, test_model_input, train_labels = df.get_input(train, test) 

train_model = MMOE(dnn_feature_columns, num_tasks=3, num_experts=7, dnn_hidden_units=(256,128),
                                   tasks=['binary', 'binary', 'binary'], device=device)
train_model.compile("adagrad", loss='binary_crossentropy')
for epoch in range(2):
	history = train_model.fit(train_model_input, train_labels, batch_size=64, epochs=4, verbose=1)
test_pred_ans = train_model.predict(test_model_input, batch_size=512) 
test_pred_ans = test_pred_ans.transpose()
df.save_result(test_pred_ans, test, 'res')

res = evaluation(data, 'res',  ['ibs', 'thyroid', 'migraine'])
```

## Supplementary
The encoding of host variables and some additional charts are provided in the supplementary files.

## Citation


## Contact
All problems please contact Meta-Spec development team: 
**Shunyao Wu**&nbsp;&nbsp;&nbsp;&nbsp;Email: wushunyao@qdu.edu.cn
