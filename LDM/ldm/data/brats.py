import os
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset

keys = ["t1n", "t1c", "t2w", "t2f"]
crop_size = (160, 160, 128)
# crop_size = (64, 64, 64)

brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=keys, allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        transforms.Lambdad(keys=keys, func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=keys),
        transforms.EnsureTyped(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys=keys, source_key="t1n", allow_missing_keys=True),
        transforms.SpatialPadd(keys=keys, spatial_size=crop_size, allow_missing_keys=True),
        transforms.RandSpatialCropd( keys=keys,
            roi_size=crop_size,
            random_center=True, 
            random_size=False,
        ),
        transforms.ScaleIntensityRangePercentilesd(keys=keys, lower=0, upper=99.75, b_min=0, b_max=1),
    ]
)

def get_brats_dataset(data_path):
    transform = brats_transforms 
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1 = os.path.join(sub_path, f"{subject}-t1n.nii.gz") 
        t1ce = os.path.join(sub_path, f"{subject}-t1c.nii.gz") 
        t2 = os.path.join(sub_path, f"{subject}-t2w.nii.gz") 
        flair = os.path.join(sub_path, f"{subject}-t2f.nii.gz") 

        data.append({"t1n":t1, "t1c":t1ce, "t2w":t2, "t2f":flair, "subject_id": subject})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)




class CustomBase(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data = get_brats_dataset(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CustomTrain(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)


class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)