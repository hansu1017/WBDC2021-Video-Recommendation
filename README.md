# **Source Codes for DDMIFM**

## **1. Environment Configuration**

- torch==1.9.0+cu111
- deepctr-torch==0.2.7
- shap==0.35.0


## **2. Catalog Structure**

```
./
├── README.md
├── Figures
├── Codes
│   ├── data
│   ├── inference
 |       ├──host_variable_features.ipynb
 |       ├──asv_only_rf.ipynb
 |       ├──asv_only_lgb.ipynb
 |       ├──asv_only_ddmifm.ipynb
 |       ├──asv_host_ddmifm.ipynb
│   ├── layer
 |       ├──MMOELayer.py
│   ├── model
 |       ├──AutomaticWeightedLoss.py
 |       ├──basemodel_captum.py
 |       ├──basemodel_uncertain.py
 |       ├──mmoe_ifm_captum_asv.py
 |       ├──mmoe_ifm_ew.py
│   ├── shap
 |       ├──shap_plot_first.ipynb
 |       ├──shap_plot_second.ipynb
 |       ├──shap_plot_third.ipynb
│   ├── utility
 |       ├──dataset.py
 |       ├──eval_mmoe_train.py
 |       ├──shap_prepare.py
```


## **3. Running Process**
- Install the environment
- Run Codes/inference/host_variable_features.ipynb to generate host_variable.csv
- Run Codes/inference/asv_only_rf.ipynb, asv_only_lgb.ipynb, asv_only_ddmifm.ipynb, asv_host_ddmifm.ipynb to obtain results of Random Forest, Lghtgbm and DDMIFM
- Run Codes/shap/shap_plot_first.ipynb, shap_plot_second.ipynb, shap_plot_third.ipynb to get data to plot feature importance
