# **Source Codes for Meta-Spec**

## Contents

- [Introduction](#introduction)
- [Package requirement](#package-requirement)
- [Pre-computing](#pre-computing)
- [Dataset](#dataset)
- [Input and output format](#input-and-output-format)
- [Run example](#run-example)
- [Run experiment](#run-experiment)
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
The ASV of 16S rRNA gene amplicons were analyzed by *Deblur*, and taxonomy was annotated by GreenGenes 13-8 database using *Parallel-Meta Suite*. The 16srRNA sequences and the information of ASVs are in the asv_info folder. 
The original dataset has 438,779 ASVs, to eliminate sparsity of ASVs, we performed a distribution-free independence test based on *mean variance index*, and selected out 844 ASVs.

\*Amir, A., et al., Deblur rapidly resolves single-nucleotide community sequence patterns. MSystems, 2017. 2(2): p. e00191-16.

\*Chen, Y., et al., Parallel-Meta Suite: interactive and rapid microbiome data analysis on multiple platforms. iMeta, 2022.

\*Cui, H., R. Li, and W. Zhong, Model-free feature screening for ultrahigh dimensional discriminant analysis. Journal of the American Statistical Association, 2015. 110(510): p. 630-641.

## Dataset
In data folder, we provide the datasets to reproduce the experiment results and run an example for Meta-Spec. 
The dataset AGP.csv includes 4437 healthy controls and 7502 patients with 844 ASVs and 56 host variables which are collected from *McDonald, D., et al.*.
Besides, example_train.csv and example_test.csv are the training set and test set to run an example which contain 800 and 200 samples respectively.

\* McDonald, D., et al., American gut: an open platform for citizen science microbiome research. Msystems, 2018. 3(3): p. e00031-18.

## Input  format
A single sample is a patient with ASVs and host variables where the ASVs must be named begin with 'asv'.
| **SampleID** | **asv1**  /**age**|
| ----------------- | --------- |
| 1             | 0.1        |1      |
| 2             | 0.05      |3      |
| 3             | 0.07      |2      |

## Run example

To run Meta-Spec, you need to run train_model.py, the auc and f1-scores will be print and the results will be saved in the current folder.
```
cd Codes/inference/
python train_model.py train_path test_path diseases_name n_expert hidden_unit1 hidden_unit2  is_print_evaluation
```
For example:
```
python train_model.py '../data/example_train.csv' '../data/example_test.csv' 'ibs thyroid migraine autoimmune lung_disease' 7 128 64 True
```
Then you can generate the bar plot of each disease by the following commends. The figures will also be saved in the current folder.
```
python train_shap.py train_path test_path diseases_name n_expert hidden_unit1 hidden_unit2 host_variables_num
python plot_shap.py train_path test_path diseases_name host_variables_num asv_num sample_num max_bar
```
For example:
```
python train_shap.py '../data/example_train.csv' '../data/example_test.csv' 'ibs thyroid migraine autoimmune lung_disease' 7 128 64 56
python plot_shap.py '../data/example_test.csv' 'ibs thyroid migraine autoimmune lung_disease' 56 100 200 60
```

## Run experiment
- Create res folder to save results
```
cd Codes/experiment
mkdir res
```
- Run rf.ipynb and lgb.ipynb to obtain results of Random Forest, Lightgbm
- Run meta_spec_without_hostvar.ipynb to get results of Meta-Spec without host variables
- Run meta_spec.ipynb to get results of Meta-Spec


## Supplementary
Some supplementary materials are provided in Supplementary folder. 
Supplementary1 is the encoding of host variables and supplementary2 is the introduction for some addtional figures.

## Citation


## Contact
All problems please contact Meta-Spec development team: 
**Xiaoquan Su**&nbsp;&nbsp;&nbsp;&nbsp;Email: suxq@qdu.edu.cn
