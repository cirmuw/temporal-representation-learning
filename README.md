# temporal-representation-learning
# Temporal Representation Learning of Phenotype Trajectories for pCR Prediction in Breast Cancer
This repository provides a supplementary code for Temporal Representation Learning of Phenotype Trajectories for pCR Prediction in Breast Cancer paper. In this work we designed a representation learning method for treatment/dissease progression learning. 

## Method
![image](https://github.com/user-attachments/assets/c4cd3cf9-53ce-4bf5-9ecf-91feb9e2df6f) 

Multi-task representation learning balances reconstruction performance (L_Rec) with temporal continuity of trajectories (L_Temp) and alignment of changes in responders (L_Align). A U-shaped denoising network extracts multi-scale features via its encoder. An MTAN-inspired masking module is used to steer attention across these tasks. The resulting trajectory representations are utilised for predicting pCR using a linear classifier. To see the integration of the losses please refer to `/utils/pretraining_engine.py`

## Data
The dataset used is a subset of the ISPY-2 cohort. Please see `/data/data_splits.txt` for the full list of patient IDs used. 
