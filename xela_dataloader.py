import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import deepdish as dd
from PIL import Image
import csv
import numpy as np
from time import sleep
import cv2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# class MySampler(torch.utils.data.Sampler):
#     def __init__(self, end_idx, seq_length):
#         indices = []
#         for i in range(len(end_idx) - 1):
#             start = end_idx[i]
#             end = end_idx[i + 1] - seq_length
#             indices.append(torch.arange(start, end))
#         indices = torch.cat(indices)
#         self.indices = indices
#
#     def __iter__(self):
#         indices = self.indices[torch.randperm(len(self.indices))]
#         return iter(indices.tolist())
#
#     def __len__(self):
#         return len(self.indices)
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(
        label_encoder.transform(list).reshape(-1, 1)
    ).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


def load_data(path, clas, init, length, log):
    cases = []
    caselist = os.listdir(os.path.join(path, clas))

    for case in caselist:
        pathTemp_visual = os.path.join(path, clas, case, "visual")
        pathTemp_tactile = os.path.join(path, clas, case, "tactile")

        v_time_path = os.path.join(pathTemp_visual, "visual_time_list.npy")
        t_time_path = os.path.join(pathTemp_tactile, "tactile_time_list.npy")

        try:
            time_list_visual = np.load(v_time_path)
            time_lst_tactile = np.load(t_time_path)
        except Exception as e:
            print(f"[SKIP] npy load failed: {clas}/{case}  err={e}")
            continue

        # ★ ここが今回のエラー対策：空ならスキップ
        if time_list_visual.size == 0:
            print(
                f"[SKIP] visual_time_list is EMPTY: {clas}/{case}  path={v_time_path}"
            )
            continue

        # フレーム数（画像枚数）は jpg を基準に数える（npy とズレる場合がある）
        visual_imgs = sorted(glob.glob(os.path.join(pathTemp_visual, "*.jpg")))
        tactile_imgs = sorted(glob.glob(os.path.join(pathTemp_tactile, "*.jpg")))
        num_visual = len(visual_imgs)
        num_tactile = len(tactile_imgs)

        # init/length 的に成立しないケースはスキップ
        if num_visual <= init + length:
            print(
                f"[SKIP] too short visual seq: {clas}/{case} num_visual={num_visual} init={init} length={length}"
            )
            continue
        if time_list_visual.shape[0] <= init + length:
            print(
                f"[SKIP] time_list_visual too short: {clas}/{case} len(time)={len(time_list_visual)} init={init} length={length}"
            )
            continue

        width, force, label = case.split("_")

        for i in range(init, num_visual - length - 1, log):
            rowTemp = [width, force, label]

            # visual paths
            for k in range(length):
                rowTemp.append(os.path.join(pathTemp_visual, f"{i+k}.jpg"))

            # tactile paths in time window
            tactile_time_length = 0
            for j in range(min(num_tactile, len(time_lst_tactile))):
                if (
                    time_lst_tactile[j] > time_list_visual[i]
                    and time_lst_tactile[j] < time_list_visual[i + length]
                ):
                    rowTemp.append(os.path.join(pathTemp_tactile, f"{j}.jpg"))
                    tactile_time_length += 1

            rowTemp.append(tactile_time_length)
            cases.append(rowTemp)

    return cases


DEFAULT_INIT = {
    "appbox": 10,
    "baisui": 10,
    "bingho": 10,
    "cesbon": 10,
    "cokele": 5,
    "haitun": 5,
    "jianjo": 5,
    "nongf1": 5,
    "pacup1": 5,
    "pacup2": 5,
    "songsu": 7,
    "zhijin": 4,
}


def train_test_dataset(
    path, visual_seq_length, tactile_seq_length, log, flag, train_classes, test_classes
):

    def _load_many(class_list):
        all_cases = []
        for clas in class_list:
            init = DEFAULT_INIT.get(clas, 5)
            all_cases += load_data(
                path, clas, init=init, length=visual_seq_length, log=log
            )
        return all_cases

    train_dataset = _load_many(train_classes)
    test_dataset = _load_many(test_classes)

    if flag == "train":
        return train_dataset
    elif flag == "test":
        return test_dataset
    else:
        raise ValueError(f"Unknown flag: {flag}")


"""
def train_test_dataset(path, visual_seq_length, tactile_seq_length, log, flag):
    appbox = load_data(path, "appbox", 10, visual_seq_length, log)
    baisui = load_data(path, "baisui", 10, visual_seq_length, log)
    bingho = load_data(path, "bingho", 10, visual_seq_length, log)
    cesbon = load_data(path, "cesbon", 10, visual_seq_length, log)
    cokele = load_data(path, "cokele", 5, visual_seq_length, log)
    haitun = load_data(path, "jianjo", 5, visual_seq_length, log)
    jianjo = load_data(path, "meinad", 5, visual_seq_length, log)
    nongf1 = load_data(path, "nongf1", 5, visual_seq_length, log)
    # load_data('/workspace/csw/graspingdata','nongfu',5,5,1)
    pacup1 = load_data(path, "pacup1", 5, visual_seq_length, log)
    pacup2 = load_data(path, "pacup2", 5, visual_seq_length, log)
    songsu = load_data(path, "songsu", 7, visual_seq_length, log)
    zhijin = load_data(path, "zhijin", 4, visual_seq_length, log)
    train_dataset = (
        appbox + baisui + bingho + cokele + haitun + jianjo + pacup1 + pacup2 + zhijin
    )
    test_dataset = cesbon + nongf1 + songsu
    if flag == "train":
        dataset = train_dataset
    elif flag == "test":
        dataset = test_dataset
    return dataset
"""


class MyDataset(Dataset):
    def __init__(
        self,
        image_paths,
        visual_seq_length,
        tactile_seq_length,
        transform_v,
        transform_t,
        log,
        flag,
        train_classes=None,
        test_classes=None,
    ):
        self.image_paths = image_paths
        self.visual_seq_length = visual_seq_length
        self.tactile_seq_length = tactile_seq_length
        self.transform_v = transform_v
        self.transform_t = transform_t
        self.log = log
        self.flag = flag

        if train_classes is None:
            train_classes = [
                "appbox",
                "baisui",
                "bingho",
                "cokele",
                "haitun",
                "jianjo",
                "pacup1",
                "pacup2",
                "zhijin",
            ]
        if test_classes is None:
            test_classes = ["cesbon", "nongf1", "songsu"]

        self.dataset = train_test_dataset(
            self.image_paths,
            self.visual_seq_length,
            self.tactile_seq_length,
            log,
            flag,
            train_classes=train_classes,
            test_classes=test_classes,
        )

        self.transform_v = transform_v
        self.transform_t = transform_t
        # self.csvReader=csv.reader(open(image_paths))
        self.label = []
        self.visual_sequence = []
        self.tactile_sequence = []
        self.classes = ["0", "1", "2"]
        self.log = log
        self.flag = flag
        # self.dataset = train_test_dataset(
        #    self.image_paths, self.visual_seq_length, self.tactile_seq_length, log, flag
        # )
        # self.tactile_sequence_length=[]
        le = LabelEncoder()
        le.fit(self.classes)

        # convert category -> 1-hot
        action_category = le.transform(self.classes).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(action_category)
        for item in self.dataset:
            self.label.append(str(item[2]))
            # self.tactile_sequence_length.append(int(item[-1]))
            visual = []
            tactile = []
            for i in range(self.visual_seq_length):
                visual.append(item[3 + i])
            for j in range(self.tactile_seq_length):
                tactile.append(item[j + 3 + self.visual_seq_length])
            self.visual_sequence.append(visual)
            self.tactile_sequence.append(tactile)
        self.label = labels2cat(le, self.label)
        # print(len(self.image_sequence))

    def __getitem__(self, index):

        visuals = []
        tactiles = []
        # tactile_raw_list = []

        for i in range(self.visual_seq_length):
            visualTemp = Image.open(self.visual_sequence[index][i])
            if self.transform_v:
                visualTemp = self.transform_v(visualTemp)
            visuals.append(visualTemp.unsqueeze(1))

        ## default ###
        for j in range(self.tactile_seq_length):
            tactileTemp = Image.open(self.tactile_sequence[index][j])
            if self.transform_t:
                tactileTemp = self.transform_t(tactileTemp)
                # print(tactileTemp.shape)
            tactiles.append(tactileTemp.unsqueeze(1))

        ## get raw
        # for j in range(self.tactile_seq_length):
        #    p = self.tactile_sequence[index][j]
        #    img = Image.open(p).convert("L").resize((4, 4))
        #    tactile_raw = np.array(img, dtype=np.float32) / 255.0
        #    tactile_raw_list.append(tactile_raw)

        #    # tactileTemp = Image.open(self.tactile_sequence[index][j])
        #    tactileTemp = Image.fromarray((tactile_raw * 255).astype(np.uint8))
        #    if self.transform_t:
        #        tactileTemp = self.transform_t(tactileTemp)
        #        # print(tactileTemp.shape)
        #    tactiles.append(tactileTemp.unsqueeze(1))

        x_v = torch.cat(visuals, dim=1)
        x_t = torch.cat(tactiles, dim=1)
        # print(x_v.shape,x_t.shape)

        y = torch.tensor(self.label[index], dtype=torch.long)
        # print(x_v.shape,x_t.shape,y)

        # tactile_raw_seq = np.stack(tactile_raw_list, axis=0)

        # return x_v, x_t, y, tactile_raw_seq
        return x_v, x_t, y

    def __len__(self):
        return len(self.visual_sequence)
