import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

HUGGINGFACE_REPO = "thuml/Time-Series-Library"
VALIDATION_SPLIT_RATIO = 0.8


class BaseSegLoader(Dataset):
    def __init__(self, train_data, test_data, test_labels, win_size, step, flag):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.train = train_data
        self.test = test_data
        self.test_labels = test_labels
        data_len = len(self.train)
        self.val = self.train[int(data_len * VALIDATION_SPLIT_RATIO):]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        else:
            i = index // self.step * self.win_size
            return np.float32(self.test[i:i + self.win_size]), np.float32(self.test_labels[i:i + self.win_size])


class PSMSegLoader(BaseSegLoader):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            ds_data = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df = ds_data["train"].to_pandas()
            test_df = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        scaler = StandardScaler()
        train_data = np.nan_to_num(train_df.values[:, 1:])
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)

        test_data = np.nan_to_num(test_df.values[:, 1:])
        test_data = scaler.transform(test_data)
        test_labels = test_label_df.values[:, 1:]
        super().__init__(train_data, test_data, test_labels, win_size, step, flag)


class MSLSegLoader(BaseSegLoader):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy", repo_type="dataset")
            test_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy", repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy", repo_type="dataset")

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(label_path)

        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        super().__init__(train_data, test_data, test_labels, win_size, step, flag)


class SMAPSegLoader(BaseSegLoader):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy", repo_type="dataset")
            test_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy", repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy", repo_type="dataset")

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(label_path)

        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        super().__init__(train_data, test_data, test_labels, win_size, step, flag)


class SMDSegLoader(BaseSegLoader):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy", repo_type="dataset")
            test_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy", repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy", repo_type="dataset")

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(label_path)

        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        super().__init__(train_data, test_data, test_labels, win_size, step, flag)


class SWATSegLoader(BaseSegLoader):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        train_path = os.path.join(root_path, "swat_train2.csv")
        test_path = os.path.join(root_path, "swat2.csv")
        if all(os.path.exists(p) for p in [train_path, test_path]):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
            train_df = ds["train"].to_pandas()
            test_df = ds["test"].to_pandas()

        test_labels = test_df.values[:, -1:]
        train_data = train_df.values[:, :-1]
        test_data = test_df.values[:, :-1]

        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        super().__init__(train_data, test_data, test_labels, win_size, step, flag)
