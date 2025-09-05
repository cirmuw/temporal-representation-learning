# temporal-representation-learning
# Temporal Representation Learning of Phenotype Trajectories for pCR Prediction in Breast Cancer
This repository provides a supplementary code for Temporal Representation Learning of Phenotype Trajectories for pCR Prediction in Breast Cancer paper. In this work we designed a representation learning method for treatment/dissease progression learning. 

## Method
![image](https://github.com/user-attachments/assets/c4cd3cf9-53ce-4bf5-9ecf-91feb9e2df6f) 

Multi-task representation learning balances reconstruction performance (L_Rec) with temporal continuity of trajectories (L_Temp) and alignment of changes in responders (L_Align). A U-shaped denoising network extracts multi-scale features via its encoder. An MTAN-inspired masking module is used to steer attention across these tasks. The resulting trajectory representations are utilised for predicting pCR using a linear classifier. To see the integration of the losses please refer to `/utils/ARTLoss.py`. 

## Data
The dataset used is a subset of 585 patients from the ISPY-2 cohort. Please see `/data/ispy2_subset_ids.txt` for the full list of patient IDs used. For each patient and each timepoint we use three DCE-derived images from ISPY-2 dataset:
* early enhancement (PE_early, 120–150 sec post-contrast)
* late enhancement (PE_late, ∼450 sec)
* signal enhancement ratio (SER = PE_early / PE_late)
  
To reduce memory usage, we generated axial-plane maximum intensity projections (MIPs) of the three DCE-derived volumes. 
![image](https://github.com/user-attachments/assets/f65a8d91-e39e-431f-b278-e6069ae8c7ee)
