import torch
import pathlib
import h5py

import pandas as pd
import numpy as np

import random

KORN_TO_LABEL = {
    "Hvete": 0,
    "Bygg": 1,
    "Havre": 2
}


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, pred=False):
        self.datapath = pathlib.Path(datapath)
        self.pred = pred

        if not self.pred:
            self.labels = self.load_labels()
        self.images = self.get_image_dict()


        self.samples = sorted(list(self.images.keys()))
        self.samples = self.samples[:50]


    def load_labels(self):
        csvs = list(self.datapath.glob("*.csv"))
        label_file = csvs[0]
        csv = pd.read_csv(label_file)
        label_dict = {}
        for i, row in csv.iterrows():
            if row["Orgnr"] not in label_dict:
                label_dict[row["Orgnr"]] = {}
            else:
                label_dict[row["Orgnr"]][row["År"]] = row["Kornsort"]

        return label_dict

    def get_image(self, orgnr, year, teigid, period):
        label = f"images/{orgnr}/{year}/{teigid}/{period}"
        with h5py.File(self.file_name, "r") as file:
            try:
                img = file[label][:]
            except Exception as e:
                return None

            img = img / 100_000_000
            for i in range(img.shape[-1]):
                min_val = np.min(img[:, :, i])
                max_val = np.max(img[:, :, i])
                if max_val != min_val:  # Prevent division by zero
                    img[:, :, i] = (img[:, :, i] - min_val) / (max_val - min_val)
                else:  # Handle case where all pixels have same value
                    if min_val > 1.0:
                        img[:, :, i] = 1.0
                    elif max_val < 0.0:
                        img[:, :, i] = 0.0
                    else:
                        img[:, :, i] = min_val
            return np.asarray(img, dtype=np.float32)

    def get_image_dict(self):
        h5s = list(self.datapath.glob("*.h5"))
        image_file = h5s[0]

        self.file_name = image_file.__str__()

        image_dict = {}
        self.teig_to_org = {}

        with h5py.File(image_file.__str__(), "r") as f:
            h5images = f['images']
            orgnrs = list(h5images.keys())
            for key in orgnrs:

                intkey = int(key)

                if not self.pred:
                    if intkey not in self.labels:
                        continue

                års = list(h5images[key].keys())
                for år in års:

                    intår = int(år)
                    if not self.pred:
                        if intår not in self.labels[intkey] or intår > 2020:
                            continue

                    if år not in image_dict:
                        image_dict[år] = {}

                    teigids = list(h5images[key][år].keys())
                    for teigid in teigids:

                        self.teig_to_org[teigid] = key

                        if teigid not in image_dict[år]:
                            image_dict[år][teigid] = []


                        periods = list(h5images[key][år][teigid].keys())
                        if len(periods) != 17:
                            del image_dict[år][teigid]
                            continue

                        if len(image_dict[år][teigid]) != 17:
                            for period in periods:
                                image_dict[år][teigid].append(period)
        new_image_dict = {}


        for år in image_dict.keys():
            for teigid in image_dict[år].keys():
                unique_key = f"{år}_{teigid}"
                if unique_key not in new_image_dict:
                    new_image_dict[unique_key] = []
                for period in image_dict[år][teigid]:
                    new_image_dict[unique_key].append(period)

        return new_image_dict

    def __getitem__(self, idx):
        image_sample = self.samples[idx]
        år, teigid = image_sample.split("_")
        orgnr = self.teig_to_org[teigid]
        periods = self.images[image_sample]

        images = np.zeros((len(periods), 25, 25, 12))
        for i, period in enumerate(periods):
            img = self.get_image(orgnr, år, teigid, period)
            if img is None:
                return self.__getitem__(random.randint(0, len(self.samples) - 1))
            images[i] = img

        images = torch.tensor(images, dtype=torch.float32)
        if not self.pred:
            label = self.labels[int(orgnr)][int(år)]
            label = KORN_TO_LABEL[label]
        else:
            label = torch.zeros((1))

        info = {
            "orgnr": orgnr,
            "år": år,
            "teigid": teigid,
            "periods": periods
        }

        images = images.permute(0, 3, 1, 2)

        return images, label, info

    def __len__(self):
        return len(self.samples)


class YieldDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        self.datapath = pathlib.Path(datapath)
        self.images = self.get_image_dict()
        self.samples = sorted(list(self.images.keys()))
        self.weather = self.load_weather()
        self.features, self.labels = self.load_features_and_labels()


    def load_weather(self):
        weatherpath = self.datapath / "weatherfeatures.csv"

        weather = pd.read_csv(weatherpath)

        weather_values = np.zeros((len(self.samples), 17, 7, 6))

        for sidx, sample in enumerate(self.samples):
            orgnr, år = sample.split("_")
            row = weather[(weather["orgnr"] == int(orgnr)) & (weather["År"] == int(år))]
            if row.empty:
                print("EMPTY")
                self.samples.pop(sidx)
                continue
            row_values = row.iloc[0, 1:-2].values
            row_values = row_values.reshape(6, 119)
            row_values = row_values.reshape(6, 17, 7)
            row_values = row_values.transpose(1, 2, 0)
            weather_values[sidx] = row_values

        return weather_values


    def load_features_and_labels(self):
        featurepath = self.datapath / "farmerfeatures.csv"
        features = pd.read_csv(featurepath).drop(["Unnamed: 0"], axis=1)
        features = features.sort_values("Yield (kg/daa)", ascending=False)
        features = features.drop_duplicates(subset=['orgnr', 'År'], keep='first')

        feature_values = np.zeros((len(self.samples), 5))
        labels = np.zeros((len(self.samples), 1))

        for sidx, sample in enumerate(self.samples):
            orgnr, år = sample.split("_")
            row = features[(features["orgnr"] == int(orgnr)) & (features["År"] == int(år))]
            if row.empty:
                self.samples.pop(sidx)
                self.weather = np.delete(self.weather, sidx, axis=0)
                continue
            row_values = row.iloc[0, 2:-1].values
            feature_values[sidx] = np.asarray(row_values)
            labels[sidx] = row.iloc[0, -1]

        return feature_values, labels

    def get_image(self, orgnr, year):
        label = f"images/{orgnr}/{year}"
        with h5py.File(self.file_name, "r") as file:
            try:
                img = file[label][:]
            except Exception as e:
                return None

            img = img / 100_000_000

            # img = (img - np.min(img)) / (np.max(img) - np.min(img))
            for i in range(img.shape[-1]):
                min_val = np.min(img[:, :, i])
                max_val = np.max(img[:, :, i])
                if max_val != min_val:  # Prevent division by zero
                    img[:, :, i] = (img[:, :, i] - min_val) / (max_val - min_val)
                else:  # Handle case where all pixels have same value
                    if min_val > 1.0:
                        img[:, :, i] = 1.0
                    elif max_val < 0.0:
                        img[:, :, i] = 0.0
                    else:
                        img[:, :, i] = min_val

            return np.asarray(img, dtype=np.float32)
    def get_image_dict(self):
        h5s = list(self.datapath.glob("*.h5"))
        image_file = h5s[0]

        self.file_name = image_file.__str__()

        image_dict = {}
        self.teig_to_org = {}

        with h5py.File(image_file.__str__(), "r") as f:
            h5images = f['images']
            orgnrs = list(h5images.keys())
            for key in orgnrs:

                if key not in image_dict:
                    image_dict[key] = []

                års = list(h5images[key].keys())

                for år in års:

                    if år not in image_dict[key]:
                        image_dict[key].append(år)

        new_image_dict = {}

        for orgnr in image_dict.keys():
            for år in image_dict[orgnr]:

                unique_key = f"{orgnr}_{år}"

                if unique_key not in new_image_dict:
                    new_image_dict[unique_key] = []

        return new_image_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_sample = self.samples[idx]
        orgnr, år = image_sample.split("_")

        # images = np.zeros((len(periods), 100, 100, 13))

        image = self.get_image(orgnr, år)[:, :, :, :-1]
        weather_features = self.weather[idx]
        features = self.features[idx]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32)
        weather_features = torch.tensor(weather_features, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        image = image.permute(0, 3, 1, 2)
        weather_features = weather_features.permute(2, 0, 1).flatten(1, 2)

        info = {
            "orgnr": orgnr,
            "år": år,
        }

        return image, weather_features, features, label, info

def get_crop_dataset_loader(datapath, batch_size, shuffle=True, pred=False):
    ds = CropDataset(datapath, pred=pred)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def get_yield_dataset_loader(datapath, batch_size, shuffle=True):
    ds = YieldDataset(datapath)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    ds = YieldDataset("data/yield/train")
    print(len(ds))
    images, weather, features, label, info = ds[0]
    print(images.shape)
    print(weather.shape)
    print(features.shape)
    print(label)
    print(info)

    print(images.min(), images.max())

    loader = get_yield_dataset_loader("data/yield/train", 8)

    for batch in loader:
        images, weather, feature, label, info = batch
        print(images.shape)
        print(weather.shape)
        print(feature.shape)
        print(label)
        print(info)
        break