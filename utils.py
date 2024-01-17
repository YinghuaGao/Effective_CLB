import random
import numpy as np
import os
import torch
import pickle

def set_random_seed(seed = 10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MyDataset(torch.utils.data.Dataset):
   
    def __init__(self, data, transform=None):

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample, label, is_poison = self.data[idx][0], self.data[idx][1], self.data[idx][2]

        if self.transform:
            sample = self.transform(sample)

        return (sample, label, is_poison)

def Add_Test_Trigger(dataset, trigger, target, alpha):
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue   
        img = (1-alpha)*img + alpha*trigger
        img = torch.clamp(img, 0, 1)
        dataset_.append((img, target))
    return dataset_

def Add_Clean_Label_Train_Trigger(dataset, trigger, target, alpha, class_order):
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            img = (1-alpha)*img + alpha*trigger
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, label, 1))
        else:
            dataset_.append((img, data[1], 0))           
    return dataset_

def get_stats(selection, output_dir, epoch, seed):
    if selection == 'loss':
        fname = os.path.join(output_dir, 'resnet_loss_grad_epoch_{}_seed_{}.pkl'.format(epoch, seed))
        with open(fname, "rb") as fin:
            loaded = pickle.load(fin)    
        stats_metric = loaded['loss']
        stats_class = loaded['class']
        stats_order = np.arange(len(stats_metric))
    elif selection == 'grad':
        fname = os.path.join(output_dir, 'resnet_loss_grad_epoch_{}_seed_{}.pkl'.format(epoch, seed))
        with open(fname, "rb") as fin:
            loaded = pickle.load(fin)    
        stats_metric = loaded['grad_norm']
        stats_class = loaded['class']
        stats_order = np.arange(len(stats_metric))
    elif selection == 'forget':
        fname = os.path.join(output_dir, 'stats_forget_seed_{}.pkl'.format(seed))
        with open(fname, 'rb') as fin:
            loaded = pickle.load(fin)
        stats_metric = loaded['forget']
        stats_class = loaded['class']
        stats_order = loaded['original_index']
    else:
        raise ValueError('Unknown selection {}'.format(selection))
    return stats_metric, stats_class, stats_order