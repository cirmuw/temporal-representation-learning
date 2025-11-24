from monai.transforms import (Compose, 
                              RandGibbsNoise, 
                              RandGaussianNoise, 
                              RandGaussianSmooth, 
                              RandAffine, 
                              ScaleIntensity)
import pickle
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from data.Dataset import ISPY2MIPParametric
import random
from utils.set_determinism import seed_all, seed_worker

# Initialize the seed for reproducibility
seed_value = 14
seed_all(seed_value)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, transform_1, trnasform_2):
        self.transform_1 = transform_1
        self.transform_2 = trnasform_2

    def __call__(self, x):
        q = self.transform_1(x.clone())
        k = self.transform_2(x.clone())

        return [q, k]
    

def get_train_dataloaders(data_path, batch_size, system='msc', timepoints=['T0','T1','T2','T3']):
    augmentation_1 = Compose([
        RandGibbsNoise(prob=1.0, alpha=(0.2,0.9)).set_random_state(seed=seed_value),
        RandGaussianNoise(prob=1.0, mean=0.5, std=0.1).set_random_state(seed=seed_value),
        ScaleIntensity(minv=0.0, maxv=1.0)

    ])
    augmentation_2 = Compose([
        RandGaussianSmooth(prob=1.0, sigma_x=(0.8, 1.5)).set_random_state(seed=seed_value),
        RandAffine(prob=1.0, shear_range=[-0.3,0.3]).set_random_state(seed=seed_value),
        ScaleIntensity(minv=0.0, maxv=1.0)
        ])
    
    with open(data_path + f'train_{system}.pkl', 'rb') as f:
        train = pickle.load(f)
        
    
    train_ds = ISPY2MIPParametric(
        train, 
        augmentation=TwoCropsTransform(augmentation_1, augmentation_2), 
        timepoints=timepoints
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler = ImbalancedDatasetSampler(train_ds),drop_last=True, worker_init_fn=seed_worker)

    return train_loader


def get_val_dataloaders(data_path, val_batch_size, system='msc', timepoints=['T0','T1','T2','T3']):
    with open(data_path + f'eval_{system}.pkl', 'rb') as f:
        val = pickle.load(f)

            
    eval_ds = ISPY2MIPParametric(val, timepoints=timepoints)
    val_loader = DataLoader(eval_ds, batch_size=val_batch_size, sampler = ImbalancedDatasetSampler(eval_ds), shuffle=False, worker_init_fn=seed_worker)

    return val_loader


def get_test_dataloaders(data_path, test_batch_size, system='msc', timepoints=['T0']):
    with open(data_path + f'test_{system}.pkl', 'rb') as f:
        test = pickle.load(f)

    
    t = list(test.items())
    random.shuffle(t)
    test = dict(t)

    test_ds = ISPY2MIPParametric(test, timepoints=timepoints)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size,  drop_last=False, worker_init_fn=seed_worker)

    return test_loader
