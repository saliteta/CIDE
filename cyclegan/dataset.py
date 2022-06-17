from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class KvairVCEDataset(Dataset):
    def __init__(self, root_Kvair, root_VCE, transform = None) -> None:
        super().__init__()
        self.root_Kvair = root_Kvair
        self.root_VCE = root_VCE
        self.transform = transform

        self.Kvair_images = os.listdir(root_Kvair) #only return the name of the jpg file
        self.VCE_images = os.listdir(root_VCE)
        self.length_dataset = max(len(self.Kvair_images), len(self.VCE_images))
        self.Kvair_len = len(self.Kvair_images)
        self.VCE_len = len(self.VCE_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        Kvair_img = self.Kvair_images[index%self.Kvair_len]
        VCE_img = self.VCE_images[index%self.VCE_len]

        Kvair_path = os.path.join(self.root_Kvair, Kvair_img)
        VCE_path = os.path.join(self.root_VCE, VCE_img)

        Kvair_img = np.array(Image.open(Kvair_path).convert("RGB")) # OPEN THE IMG
        VCE_img = np.array(Image.open(VCE_path).convert("RGB")) # OPEN THE IMG

        if self.transform:
            augmentations = self.transform(image = Kvair_img, image0 = VCE_img)
            Kvair_img = augmentations["image"]
            VCE_img = augmentations["image0"]

        return Kvair_img, VCE_img
