from torch.utils.data import DataLoader,Subset,ConcatDataset
from torchvision.transforms import *
from torchvision.datasets import CIFAR100
from torch.nn import CrossEntropyLoss,Softmax
from torch.optim import *
from dataset import PUFImageDataset
from tqdm import tqdm,trange
from model import *
import random,torch
import numpy as np
import pandas as pd

class PUF_VGG16_n:
    """
    original VGG16
    """
    def __init__(self,path_prefix='/home/sunyf23/Work_station/PUF_Phenotype/Latency-DRAM-PUF-Dataset',seed=5,k_fold=5,transform=None,target_transform=None,device_num=0) -> None:
        if torch.cuda.is_available():
            self.device = 'cuda'
            try:
                torch.cuda.set_device(device_num)
            except:
                torch.cuda.set_device(0)
        else:
            self.device = 'cpu'
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)        
        self.k_fold = k_fold
        
        
        transform = Compose([ToTensor(),Lambda(lambda x: x.to(self.device))])
        # target_transform=Lambda(lambda y: (torch.zeros(100, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)).to(self.device))
        target_transform=Lambda(lambda y: torch.tensor(y).to(self.device))
        self.dataset_all = CIFAR100(root="data",train=True,transform=transform,target_transform=target_transform,download=True)
                
        self.model = VGG16(num_classes=100).to(self.device)
        
        total_params = 0
        for param in list(self.model.parameters()):
            nn = 1
            for sp in list(param.size()):
                nn = nn * sp
            total_params += nn
        self.total_params = total_params
        print("Total parameters", self.total_params)
        model_params = filter(lambda param: param.requires_grad,
                              self.model.parameters())
        trainable_params = sum([np.prod(param.size())
                                for param in model_params])
        self.trainable_params = trainable_params
        print("Trainable parameters", self.trainable_params)

        self.criterion = 'ce'
        self.epochs = 20
        self.optimizer = 'adam'
        self.learning_rate = 1e-3
        self.weight_decay = 0

    def train_k_fold(self):
        """
        k_fold the dataset and train each fold
        """
        indices = list(range(len(self.dataset_all)))
        random.shuffle(indices)
        sub_len = len(indices)//self.k_fold

        k_set = []
        for i in range(self.k_fold-1):
            k_set.append(indices[sub_len*i:sub_len*(i+1)])
        k_set.append(indices[sub_len*(self.k_fold-1):])

        all_index = set(indices)
        for i in trange(self.k_fold,desc=f'{self.k_fold} Folds',leave=True,position=0):
            test_index = set(k_set[i])
            train_index = all_index - test_index
            train_dataset = Subset(self.dataset_all,list(train_index))
            test_dataset = Subset(self.dataset_all,list(test_index))
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)
            # print(len(train_dataloader))
            test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=False)
            model = self.train(train_dataloader)
            # self.test(test_dataloader,model)

    def train(self,train_dataloader):
        model = self.model

        if self.criterion == 'ce':
            loss_fn = CrossEntropyLoss()
        if self.optimizer == 'adam':
            optimizer = torch.optim.SGD(model.parameters(), 
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
            
        
        size = len(train_dataloader)
        train_bar = tqdm(range(self.epochs),position=1,leave=False,colour='#3399FF',desc=f"{self.epochs} Epochs")
        for T in train_bar:
            model.train()
            batch_bar = trange(size,position=2,leave=False,colour='#33CC00')
            # batch_bar = tqdm(train_dataloader,position=2,leave=False,colour='#33CC00')
            for _,(X, y) in zip(batch_bar,train_dataloader):
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                batch_bar.set_description(f"Loss: {loss:>7f}")
            

        return model

    def test(self,dataloader,model):
        model.eval()
        if self.criterion == 'ce':
            loss_fn = CrossEntropyLoss()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                # print(y.size())
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"\nTest Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")        

        

if __name__ == '__main__':
    
    model = PUF_VGG16_n()
    model.train_k_fold()