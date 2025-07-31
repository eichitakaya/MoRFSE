import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MoRFSEDataset(Dataset):
    def __init__(self, csv_file_path, transform=None, network="gating", mode="train", num_test_fold=0):
        self.csv_file_path = csv_file_path
        self.transform = transform
        self.network = network
        self.mode = mode
        self.num_test_fold = num_test_fold

        self.num_class = 3

        self.data = []
        self.img_paths = []
        self.labels = []
        
        df = pd.read_csv(self.csv_file_path)
        
        if mode == "train":
            df = df.query(f"fold != {self.num_test_fold}")
            
            if self.network == "gating":
                df = df.query("lesion_type == 0 or lesion_type == 1")
                self.img_paths = df["img_path"].tolist()
                self.labels = df["lesion_type"].tolist()
                
            elif self.network == "expert1":
                df = df.query("lesion_type == 0 or lesion_type == 2")
                self.img_paths = df["img_path"].tolist()
                self.labels = df["lesion_category"].tolist()
                
            elif self.network == "expert2":
                df = df.query("lesion_type == 1 or lesion_type == 2")
                self.img_paths = df["img_path"].tolist()
                self.labels = df["lesion_category"].tolist()
        
        else:
            df = df.query(f"fold == {self.num_test_fold}")
        
            self.img_paths = df["img_path"].tolist()
            self.labels = df["lesion_category"].tolist()

        self.data = list(zip(self.img_paths, self.labels))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_basename = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, img_basename