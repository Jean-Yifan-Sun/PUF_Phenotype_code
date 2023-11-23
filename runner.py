from torch.utils.data import DataLoader,Subset,ConcatDataset,random_split
from torchvision.transforms import *
from torchmetrics import Accuracy,F1Score
from torch.nn import CrossEntropyLoss,Softmax
from torch.optim import *
from dataset import PUFImageDataset
from tqdm import tqdm,trange
from model import *
from settings import Settings
import random,torch,os
import numpy as np
import pandas as pd

class PUF_VGG16_n:
    """
    original VGG16
    """
    def __init__(self) -> None:
        ss = Settings()
        self.ss = ss
        self.device_num = ss.args.device_num
        self.seed = ss.args.seed
        self.k_fold = ss.args.k_fold
        self.model_type = ss.args.model_type
        self.criterion = ss.args.criterion
        self.epochs = ss.args.epochs
        self.optimizer = ss.args.optim_type
        self.learning_rate = ss.args.learning_rate
        self.weight_decay = ss.args.weight_decay
        self.test_frac = ss.args.test_frac
        self.momentum = ss.args.momentum
        self.output_path = ss.args.output_path
        path_prefix = ss.args.path_prefix

        if torch.cuda.is_available():
            self.device = 'cuda'
            try:
                torch.cuda.set_device(self.device_num)
            except:
                torch.cuda.set_device(0)
        else:
            self.device = 'cpu'
        
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)        
        
        self.mean,self.std = PUFImageDataset(path_prefix=path_prefix,transform=None,target_transform=None).mean_std()
        transform = Compose([Normalize(self.mean,self.std),Lambda(lambda x: x.to(self.device))])
        target_transform=Lambda(lambda y: (torch.zeros(5, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)).to(self.device))
        # target_transform=Lambda(lambda y: torch.tensor(y).to(self.device))
        self.dataset_all = PUFImageDataset(path_prefix=path_prefix,transform=transform,target_transform=target_transform)
        self.Acc = Accuracy(task='multiclass',num_classes=self.dataset_all.num_class).to(self.device)   
        self.F1 = F1Score(task='multiclass',num_classes=self.dataset_all.num_class).to(self.device)
        self.test_result = {}

    def train_k_fold(self):
        """
        k_fold the dataset and train each fold
        """
        self.model_list = []
        generator = torch.Generator().manual_seed(self.seed)
        self.test_set,self.train_set = random_split(self.dataset_all,[self.test_frac,1-self.test_frac],generator=generator)
        self.test_loader = DataLoader(self.test_set, batch_size=64, shuffle=True, pin_memory=False)
        print(f"\nTraining on {len(self.train_set)} samples, test on {len(self.test_set)} samples.\n",)
        indices = list(range(len(self.train_set)))
        random.shuffle(indices)
        sub_len = len(indices)//self.k_fold

        k_set = []
        for i in range(self.k_fold-1):
            k_set.append(indices[sub_len*i:sub_len*(i+1)])
        k_set.append(indices[sub_len*(self.k_fold-1):])

        all_index = set(indices)
        for i in trange(self.k_fold,desc=f'{self.k_fold} Folds',leave=False,position=0):
            test_index = set(k_set[i])
            train_index = all_index - test_index
            train_dataset = Subset(self.train_set,list(train_index))
            test_dataset = Subset(self.train_set,list(test_index))
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)
            # print(len(train_dataloader))
            test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=False)
            model = self.train(train_dataloader)
            self.model_list.append(model)
            self.test(test_dataloader,model,mode=f'k{i+1}_test')
            self.test(self.test_loader,model,mode=f'test_k{i+1}')

        self.test_k_fold()

    def train(self,train_dataloader):
        if  self.model_type == 'vgg16':
            model = VGG16(num_classes=self.dataset_all.num_class).to(self.device)
            total_params = 0
            for param in list(model.parameters()):
                nn = 1
                for sp in list(param.size()):
                    nn = nn * sp
                total_params += nn
            self.total_params = total_params
            print("Total parameters", self.total_params)
            model_params = filter(lambda param: param.requires_grad,
                                model.parameters())
            trainable_params = sum([np.prod(param.size())
                                    for param in model_params])
            self.trainable_params = trainable_params
            print("Trainable parameters", self.trainable_params)
        if self.criterion == 'ce':
            loss_fn = CrossEntropyLoss()
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), 
                                         lr=self.learning_rate,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), 
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
            
        
        # size = len(train_dataloader.dataset)
        train_bar = tqdm(range(self.epochs),position=1,leave=False,colour='#3399FF',desc=f"{self.epochs} Epochs")
        for T in train_bar:
            model.train()
            # batch_bar = trange(size,position=2,leave=False,colour='#33CC00')
            batch_bar = tqdm(train_dataloader,position=2,leave=False,colour='#33CC00')
            for(X, y) in batch_bar:
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                batch_bar.set_description(f"Loss: {loss:>7f}")
        
        return model

    def test(self,dataloader,model,mode):
        model.eval()
        if self.criterion == 'ce':
            loss_fn = CrossEntropyLoss()
     
        preds,ys = [], []
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                ys.append(y)
                preds.append(pred)
       
        preds = torch.concat(preds,dim=0)
        ys = torch.concat(ys,dim=0)
        # print(preds.size(),ys.size())
        acc = self.Acc(preds.argmax(1),ys.argmax(1)).item()
        test_loss = loss_fn(preds,ys).item()
        f1 = self.F1(preds.argmax(1),ys.argmax(1)).item()
        print(f"\nFold {mode} Test Result: Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f}, F1 score: {f1:>8f} \n")
        self.test_result[mode]={'acc':acc,'f1':f1,'loss':test_loss}        

    def test_k_fold(self):
        assert len(self.model_list)==self.k_fold
        if self.criterion == 'ce':
            loss_fn = CrossEntropyLoss()
        test_loader = self.test_loader
        softmax = Softmax(dim=1)
        preds,ys = [], []
        with torch.no_grad():
            for X, y in test_loader:
                pred = 0
                ys.append(y)
                for i in self.model_list:
                    i.eval()
                    pred += i(X)/self.k_fold
                preds.append(pred)
        preds = torch.concat(preds,dim=0)
        ys = torch.concat(ys,dim=0)
        acc = self.Acc(preds.argmax(1),ys.argmax(1)).item()
        test_loss = loss_fn(preds,ys).item()
        f1 = self.F1(preds.argmax(1),ys.argmax(1)).item()
        print(f"\nTest Result: Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f}, F1 score: {f1:>8f} \n")
        fold_pred = softmax(preds).detach().cpu().numpy()
        df = pd.DataFrame(np.around(fold_pred,3))
        df.columns = ['delta', 'gamma', 'epsilon', 'beta', 'alpha']
        df.to_csv(os.path.join(self.output_path,f'seed_{self.seed}/Train_Fold_model_socres.csv'))

        self.test_result['final']={'acc':acc,'f1':f1,'loss':test_loss} 
        csv_path = os.path.join(self.output_path,f'seed_{self.seed}/Test_result.csv')
        pd.DataFrame(self.test_result).to_csv(csv_path)
        for i in range(len(self.model_list)):
            pt_path = os.path.join(self.output_path,f'seed_{self.seed}/model_weights/k{i+1}_model_weights.pt')
            torch.save(self.model_list[i].state_dict(), pt_path)

        

if __name__ == '__main__':
    
    model = PUF_VGG16_n()
    model.train_k_fold()
    print(model.test_result)