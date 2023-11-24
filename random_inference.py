from torch.utils.data import DataLoader,Subset,ConcatDataset,random_split,Dataset
from torchvision.transforms import *
from torch.nn import CrossEntropyLoss,Softmax
from torchmetrics import Accuracy,F1Score
from tqdm import trange,tqdm
from model import *
from settings import Settings
import torch,os,random
import numpy as np
import pandas as pd

class Fake_dataset(Dataset):
    def __init__(self,fake_num:int,seed:int,device,transform,target_transform) -> None:
        super(Fake_dataset,self).__init__()
        self.num = fake_num
        self.seed = seed
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.num_class = 5
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.all_data = [torch.randint(low=0,high=256,size=(3, 220, 200),
                                       device=self.device,generator=self.generator,dtype=torch.float) for i in range(self.num)] 
        self.all_label = [torch.randint(low=0,high=4,size=(1,1),
                                       device=self.device,generator=self.generator,dtype=torch.int) for i in range(self.num)]
        
    def __len__(self):
        return self.num
    
    def __getitem__(self, index):
        image = self.all_data[index]
        label = self.all_label[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def mean_std(self):
        '''
        Compute mean and variance for training data
        :return: (mean, std)
        '''
        print('Compute mean and variance for training data:')
        mean = torch.zeros(3).to(self.device)
        std = torch.zeros(3).to(self.device)
        for i in range(self.__len__()):
            X, _ = self.__getitem__(i)
            for d in range(3):
                mean[d] += X[ d, :, :].mean()
                std[d] += X[ d, :, :].std()
        mean.div_(self.__len__())
        std.div_(self.__len__())
        return mean.type(torch.float),std.type(torch.float)

class Inference_Attack():
    """
    Inference Attack on VGG16
    """
    def __init__(self) -> None:
        ss = Settings()
        self.ss = ss
        if torch.cuda.is_available():
            self.device = 'cuda'
            try:
                torch.cuda.set_device(self.device_num)
            except:
                torch.cuda.set_device(0)
        else:
            self.device = 'cpu'
        
        self.criterion = ss.args.criterion
        self.device_num = ss.args.device_num
        self.seed = ss.args.seed
        self.k_fold = ss.args.k_fold
        self.model_type = ss.args.model_type
        self.model_path = os.path.join(ss.args.model_path, f'seed_{self.seed}/model_weights')
        self.fake_num = ss.args.fake_num
        self.output_path = ss.args.output_path

        self.mean,self.std = Fake_dataset(fake_num=self.fake_num,
                                          seed=self.seed,device=self.device,
                                          transform=None,target_transform=None).mean_std()
        transform = Normalize(self.mean,self.std)
        target_transform = None

        self.attack_dataset = Fake_dataset(fake_num=self.fake_num,
                                          seed=self.seed,device=self.device,
                                          transform=transform,target_transform=target_transform)
        
        self.attack_dataloader = DataLoader(self.attack_dataset,
                                            batch_size=64,shuffle=True)
        
        self.Acc = Accuracy(task='multiclass',num_classes=self.attack_dataset.num_class).to(self.device)   
        self.F1 = F1Score(task='multiclass',num_classes=self.attack_dataset.num_class).to(self.device)
        self.result = {}
        self.inference_output = []
        
    def attack(self):
        """
        compute model outputs and attack result without threshold
        """
        softmax = Softmax(dim=1)
        if self.model_type == 'vgg16':
            model = VGG16(num_classes=self.attack_dataset.num_class).to(self.device)
        if self.criterion == 'ce':
            loss_fn = CrossEntropyLoss()

        model_paths = os.listdir(self.model_path)
        for i in range(len(model_paths)):
            state_dict = torch.load(os.path.join(self.model_path,model_paths[i]))
            model.load_state_dict(state_dict)
            model.eval()
            preds,ys = [], []
            with torch.no_grad():
                for X, y in self.attack_dataloader:
                    pred = softmax(model(X))
                    ys.append(y)
                    preds.append(pred)

            preds = torch.concat(preds,dim=0)
            ys = torch.concat(ys,dim=0).squeeze()
            self.inference_output.append((preds,ys))
            acc = self.Acc(preds.argmax(1),ys).item()
            test_loss = loss_fn(preds,ys.long()).item()
            f1 = self.F1(preds.argmax(1),ys).item()
            print(f"\nAttack Test Result: k{i+1} Attack num: {self.fake_num} Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f}, F1 score: {f1:>8f} ")
            self.result[f'k{i+1}']={'attack_num':self.fake_num,'acc':acc,'f1':f1,'loss':test_loss}

            df = pd.DataFrame(np.around(preds.detach().cpu().numpy(),3))
            df.columns = ['delta', 'gamma', 'epsilon', 'beta', 'alpha']
            df.to_csv(os.path.join(self.output_path,f'seed_{self.seed}/k{i+1}_model_attack_socres.csv'))

        # pd.DataFrame(self.result).to_csv(os.path.join(self.output_path,f'seed_{self.seed}/Attack_result_no_thr.csv'))

    def get_score(self):
        """
        compute model scores of all folds and the final model
        """
        softmax = Softmax(dim=1)
        fold_pred = 0
        for i in trange(len(self.inference_output),position=0):
            preds,ys = self.inference_output[i]
            preds = softmax(preds)
            fold_pred += preds/self.k_fold
            
            df = pd.DataFrame(np.around(preds.detach().cpu().numpy(),3))
            df.columns = ['delta', 'gamma', 'epsilon', 'beta', 'alpha']
            df.to_csv(os.path.join(self.output_path,f'seed_{self.seed}/k{i+1}_model_attack_socres.csv'))
        self.inference_output.append((fold_pred,ys))
        df = pd.DataFrame(np.around(fold_pred.detach().cpu().numpy(),3))
        df.columns = ['delta', 'gamma', 'epsilon', 'beta', 'alpha']
        df.to_csv(os.path.join(self.output_path,f'seed_{self.seed}/k{len(self.inference_output)}_model_attack_socres.csv'))
    
    def get_threshold(self):
        """
        compute attack result with threshold for each fold and final model
        """
        threshold_dict = {}
        # softmax = Softmax(dim=1)
        thresholds = [i/100 for i in range(100)]
        for i in trange(len(self.inference_output),position=0):
            preds,ys = self.inference_output[i]
            # preds = softmax(preds)
            # indexs = preds.argmax(dim=1)
            threshold_dict[f'model_k{i+1}'] = {}
            for k in trange(len(thresholds),position=1,leave=False):
                count = 0
                thr = thresholds[k]
                for j in range(len(preds)):
                    pred = preds[j]
                    # ind = indexs[j].item()
                    val = pred.max().item()
                    # y = ys[j].item()
                    if val >= thr :
                        count += 1
                acc = round(count/len(preds),3)
                threshold_dict[f'model_k{i+1}'][thr]=acc
        # print(threshold_dict)
        pd.DataFrame(threshold_dict).to_csv(os.path.join(self.output_path,f'seed_{self.seed}/Attack_result_with_thr.csv'))

if __name__ == '__main__':
    attacker = Inference_Attack()
    attacker.attack()
    # attacker.get_score()
    attacker.get_threshold()