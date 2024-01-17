import argparse
import numpy as np
import numpy.random as npr
import time
import os
import pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import grad
from torchvision import datasets, transforms
from cifar_resnet import ResNet18, ResNet50
from utils import *
import logging


def train_step(args, model, device, trainset, model_optimizer, epoch, example_stats, logger):
    train_loss = 0.
    correct = 0.
    total = 0.
    model.train()
    trainset_permutation_inds = npr.permutation(np.arange(len(trainset)))
    batch_size = args.batch_size
    for batch_id, batch_start_ind in enumerate(range(0, len(trainset), batch_size)):
        if batch_start_ind + batch_size > len(trainset_permutation_inds):
            batch_inds = trainset_permutation_inds[batch_start_ind:]
        else:
            batch_inds = trainset_permutation_inds[batch_start_ind:batch_start_ind + batch_size]
        transformed_trainset = []
        trainset_targets = []
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        for ind in batch_inds:
            trainset_targets.append(trainset.__getitem__(ind)[1])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(trainset_targets)
        inputs, targets = inputs.to(device), targets.to(device)
        model_optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        acc = predicted == targets
        for j, index in enumerate(batch_inds):
        
            # Get index in original dataset (not sorted by forgetting)
            index_in_original_dataset = train_indx[index]
        
            # Save whether example was correctly classified
            index_stats = example_stats.get(index_in_original_dataset, [])
            index_stats.append(acc[j].sum().item())
            example_stats[index_in_original_dataset] = index_stats
        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss.backward()
        model_optimizer.step()

        # Add training accuracy to dict
        acc_stats = example_stats.get('train', [])
        acc_stats.append(100. * correct.item() / float(total))
        example_stats['train'] = acc_stats

    if (epoch+1)%10 == 0:
        tr_loader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
        loss_total = np.zeros(len(trainset))
        grad_norm_f_total = np.zeros(len(trainset))
        class_total = np.zeros(len(trainset))
        for ind in range(len(trainset)):
            inputs =  tr_loader.collate_fn([trainset.__getitem__(ind)[0]])
            targets = tr_loader.collate_fn([trainset.__getitem__(ind)[1]])
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs.clone()
            x.requires_grad_(True)
            model.eval()
            outputs = model(x)
            loss = criterion(outputs, targets)
            params = [ p for p in model.parameters() if p.requires_grad ]
            grad_list = list(grad(loss, params))
            y = 0
            for i in range(len(grad_list)):
                y += torch.sum(grad_list[i]**2)
            loss_total[ind] = loss.detach().cpu().numpy()[0]
            grad_norm_f_total[ind] = torch.sqrt(y).detach().cpu().numpy()
            class_total[ind] = trainset.__getitem__(ind)[1]      
        
        # Save loss and gradient norm statistics
        stats_loss_grad = {}
        stats_loss_grad['loss'] = loss_total
        stats_loss_grad['grad_norm'] = grad_norm_f_total
        stats_loss_grad['class'] = class_total
    
        fname = os.path.join(args.output_dir, 'resnet_loss_grad_epoch_{}_seed_{}.pkl'.format(epoch+1, args.seed))
        with open(fname, "wb") as f:
            pickle.dump(stats_loss_grad, f)
    
    return train_loss / (batch_id+1), correct.item() / float(total)
        

def test_step(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def compute_forgetting_statistics(diag_stats, npresentations, logger):
    unlearned_per_presentation = {}
    first_learned = {}

    for example_id, example_stats in diag_stats.items():
        if not isinstance(example_id, str):
            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats[:npresentations])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(
                    transitions == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            # Find the presentation at which the example was first learned, 
            # e.g. first presentation when acc is 1
            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

    return unlearned_per_presentation, first_learned

def sort_examples_by_forgetting(train_set, unlearned_per_presentation_all, first_learned_all, npresentations, logger):
    example_original_order = []
    example_stats = []
    train_target = []
    for example_id in unlearned_per_presentation_all.keys():
        example_original_order.append(example_id)
        example_stats.append(0)
        train_target.append(train_set[example_id][1])
        stats = unlearned_per_presentation_all[example_id]

        # If example was never learned during current training run, add max forgetting counts
        if np.isnan(first_learned_all[example_id]):
            example_stats[-1] += npresentations
        else:
            example_stats[-1] += len(stats)

    logger.info('Number of unforgettable examples: {}'.format(
        len(np.where(np.array(example_stats) == 0)[0])))
    
    return example_original_order, example_stats, train_target


parser = argparse.ArgumentParser(description='Calculate different metrics for poisoned sample selection')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--output_dir', default='save_metric', help='directory where to save results')
parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50'])

args = parser.parse_args()
logger = logging.getLogger()
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output_{}.log'.format(args.seed))),
            logging.StreamHandler()
        ])

use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True
set_random_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_indx = np.array(range(len(train_dataset)))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.model == 'resnet18':
    model = ResNet18(num_classes=10)
elif args.model == 'resnet50':
    model = ResNet50(num_classes=10)
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
criterion.__init__(reduce=False)
test_criterion = torch.nn.CrossEntropyLoss().to(device)
model_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(model_optimizer, milestones=[60, 90], gamma=0.1)

# Initialize dictionary to save statistics for every example presentation
example_stats = {}
logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t TestLoss \t TestACC')
for epoch in range(args.epochs):
    start_time = time.time()
    train_loss, train_acc = train_step(args, model, device, train_dataset, model_optimizer, epoch, example_stats, logger)
    test_loss, test_acc = test_step(model, test_criterion, test_loader)
    epoch_time = time.time() - start_time
    logger.info('%d \t %.4f \t %.2f \t %.4f \t %.4f \t %.4f \t %.4f' % (epoch, scheduler.get_lr()[0], epoch_time, train_loss, train_acc, test_loss, test_acc))
    scheduler.step()

unlearned_per_presentation, first_learned = compute_forgetting_statistics(example_stats, args.epochs, logger)
example_original_order, forget_stats,  train_target = sort_examples_by_forgetting(train_dataset, unlearned_per_presentation, 
                                                                                 first_learned, args.epochs, logger)

stats_forget = {}
stats_forget['forget'] = forget_stats
stats_forget['class'] = train_target
stats_forget['original_index'] = example_original_order
fname = os.path.join(args.output_dir, 'stats_forget_seed_{}.pkl'.format(args.seed))
with open(fname, "wb") as f:
    pickle.dump(stats_forget, f)