import torch
import nibabel as nib
from torch.utils.data import Dataset
from monai.transforms import Compose, ScaleIntensity, Resize
from tqdm import tqdm
import monai
from typing import List, Dict, Optional

class ISPY2MIPParametric(Dataset):

    def __init__(
            self, 
            data_list: Dict, 
            timepoints:Optional[List[str]] =['T0','T1','T2','T3'], 
            augmentation: monai.transforms = None, 
            res: tuple = (256,256),
            two_view_transform: bool = False
            ):
        
        self.timepoints = timepoints
        self.augmentation = augmentation
        self.preprocessing_transform = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=res),
        ])
        self.data_list = self.load_data(data_list)
        self.two_view_transform = two_view_transform

    def get_labels(self):
        labels = [d.get('pcr') for d in self.data_list]
        return labels
    
    def load_three_channel_data(self, timepoint_directory_path):
        pe_early = self.preprocessing_transform(
            torch.FloatTensor(
                nib.load(timepoint_directory_path + 'pe_early.nii.gz').get_fdata()).unsqueeze(0))
        
        pe_late = self.preprocessing_transform(
            torch.FloatTensor(
                nib.load(timepoint_directory_path + 'pe_late.nii.gz').get_fdata()).unsqueeze(0))
        ser = self.preprocessing_transform(
            torch.FloatTensor(
                nib.load(timepoint_directory_path + 'ser.nii.gz').get_fdata()).unsqueeze(0))
        
        
        return torch.stack([pe_early, pe_late, ser], dim=0).squeeze()

    def load_data(self, data_dict:Dict):
        print('loading the data ...', flush=True)
        data_list = [] # list of data dictionaries for all of the patients
        for patient in tqdm(data_dict.keys()):
            patient_dict = data_dict[patient]
            loaded_data_dict = {} # dictionary of a single patient data e.g. {'T0' : torch.FloatTensor, 'T1' ..etc}
            loaded_data_dict['id'] = patient
            loaded_data_dict['pcr'] = patient_dict.get('pcr')
            for t in self.timepoints:
                loaded_data_dict[t] = self.load_three_channel_data(patient_dict.get(t))
                loaded_data_dict[f'target_{t}'] = self.load_three_channel_data(patient_dict.get(t))
            data_list.append(loaded_data_dict)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_dict = self.data_list[idx].copy()
        if self.augmentation is not None:
            for t in self.timepoints:
                if self.two_view_transform:
                    #HACK to make it compatible with monai
                    view1, view2 = self.augmentation(data_dict.get(t))
                    #HACK to make it compatible with monai
                    if type(view1) == monai.data.meta_tensor.MetaTensor and type(view2) == monai.data.meta_tensor.MetaTensor:
                        data_dict[t] = [view1.as_tensor(), view2.as_tensor()]
                    else:
                        data_dict[t] = [view1[0].as_tensor(), view2[0].as_tensor()]
                else:
                    data_dict[t] = self.augmentation(data_dict.get(t))
        return  data_dict
