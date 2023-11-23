import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

class PUFImageDataset(Dataset):
    def __init__(self, path_prefix, transform=None, target_transform=None):
        image_path = os.path.join(path_prefix,'grayscale_images') 
        image_labels = os.listdir(image_path)
        images = []
        labels = []
        dimm_locations = []
        dimm_temperatures = []
        dimm_voltages = []
        measurement_num = []
        miss = 0
        for i in image_labels:
            locations = os.listdir(os.path.join(image_path,i))
            location_path = os.path.join(image_path,i)
            for j in locations:
                temperatures = os.listdir(os.path.join(location_path,j))
                temperature_path = os.path.join(location_path,j)
                for k in temperatures:
                    voltage_path = os.path.join(temperature_path,k)
                    voltages = os.listdir(voltage_path)
                    for v in voltages:
                        pattern_path = os.path.join(voltage_path,v)
                        patterns = os.listdir(pattern_path)
                        rgb = {}
                        for p in patterns:
                            fig_path = os.path.join(pattern_path,p)
            
                            rgb[p]={}
                            for f in range(10):
                                # delta_b_20_1.5_patm_grayscale_2_.png
                                
                                temp = os.path.join(fig_path,f'{i}_{j}_{k}_{v}_{p}_grayscale_{f+1}_.png')
                                rgb[p][f]=temp
                                # images.append(temp)
                                # labels.append(i)
                                # dimm_locations.append(j)
                                # dimm_temperatures.append(k)
                                # dimm_voltages.append(v)
                                # dimm_patterns.append(p)
                                # measurement_num.append(f)

                        for f in range(10):
                            temp = []
                            for p in patterns:
                                rgb_image = rgb[p][f]
                                if os.path.exists(rgb_image):
                                    temp.append(rgb_image)
                                else:
                                    
                                    print(f'WARNING: label {i} location {j} temperature {k} voltage {v} pattern {p} missing Nb.{f+1} pic!')
                                    miss+=1
                                    break
                            if len(temp)==3:
                                images.append(temp)
                                dimm_voltages.append(v)
                                dimm_temperatures.append(k)
                                dimm_locations.append(j)
                                labels.append(i)
                                measurement_num.append(f)
        data_dict = {
            'images':images,
            'labels':labels,
            'dimm_locations':dimm_locations,
            'dimm_temperatures':dimm_temperatures,
            'dimm_voltages':dimm_voltages,
            # 'dimm_patterns':dimm_patterns,
            'measurement_num':measurement_num

        }
        print(f"Missing total {miss} pics.")
        self.patterns = patterns
        self.locations = locations
        self.tempatures = temperatures
        self.voltages = voltages
        self.measurements = list(range(10))
        self.img_labels = image_labels
        self.num_class = len(image_labels)
        num_labels = self.num_label(labels)
        data_dict['num_labels'] = num_labels
        self.img_df = pd.DataFrame.from_dict(data_dict)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        
        return len(self.img_df)

    def __getitem__(self, idx):
        img_path = self.img_df.loc[idx,'images']
        images = []
        for i in img_path: 
            image = read_image(i).type(torch.float)
            images.append(image)
        image = torch.concat(images,dim=0)
        label = int(self.img_df.loc[idx, 'num_labels'])
        if self.transform:
                image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def num_label(self,labels):
        one_hot_labels = []
        print(f"Labels to numeric by index:{self.img_labels}")
        for i in labels:
            one_hot_labels.append(self.img_labels.index(i))
        return torch.IntTensor(one_hot_labels)

    def mean_std(self):
        '''
        Compute mean and variance for training data
        :return: (mean, std)
        '''
        print('Compute mean and variance for training data:')
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for i in range(self.__len__()):
            X, _ = self.__getitem__(i)
            for d in range(3):
                mean[d] += X[ d, :, :].mean()
                std[d] += X[ d, :, :].std()
        mean.div_(self.__len__())
        std.div_(self.__len__())
        return mean.type(torch.float),std.type(torch.float)


if __name__ == '__main__':
    dataset = PUFImageDataset(path_prefix='/home/sunyf23/Work_station/PUF_Phenotype/Latency-DRAM-PUF-Dataset')
    img_path = dataset.img_df['images']
    for i in img_path:
        temp = i
        keys = temp[0].split('/')[-1].split('_')
        for j in temp:
            
            new_keys = j.split('/')[-1].split('_')
            for index in [0,1,2,3,5,6,7]:
                assert keys[index] == new_keys[index]
    for i in ['delta','gamma','epsilon','beta','alpha']:
        lenth = len(dataset.img_df[dataset.img_df['labels']==i])
        print(f"{i}:{lenth}")
    # print(dataset.img_df['images'][0])
    # print(len(dataset.img_df))
