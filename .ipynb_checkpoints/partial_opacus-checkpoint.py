import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
from time import sleep
import numpy as np
from models.resnetDP import *
import PIL
import ipdb
import sys
from opacus import PrivacyEngine, GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants import RDPAccountant
from utils.dpaccountant import  get_sigma
from utils.CNN import *
from utils.masked_dataset import *
from utils.custom_optimizer import HybridDPOptimizer
from rdp_accountant import * 
import os
import wandb

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', default = "FMNIST")
parser.add_argument('--eps', default = 3, type=float)
parser.add_argument('--delta', default = 1e-5, type=float)
parser.add_argument('--lbeps', default = 1.5, type=float)
parser.add_argument('--lbs', default = 10, type=float)
parser.add_argument('--feat', default = 18, type=int)
parser.add_argument('--aux', default = 0.01, type=float)
parser.add_argument('--bz', default = 128, type=int)
parser.add_argument('--bzw', default = 128, type=int)
parser.add_argument('--bza', default = 128, type=int)
parser.add_argument('--lr', default = 0.01, type=float)
parser.add_argument('--lrw', default = 0.02, type=float)
parser.add_argument('--ep', default = 20, type=int)
parser.add_argument('--epw', default = 10, type=int)
parser.add_argument('--model')
parser.add_argument('--decay',default=5,type=int)

args = parser.parse_args()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameter
DEFAULT_ALPHAS = np.array([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))

require_eps = args.eps
require_delta = args.delta

label_eps = args.lbeps
label_scale = args.lbs
label_laplace_rdp = np.log(DEFAULT_ALPHAS * np.exp(label_eps*(DEFAULT_ALPHAS-1)) /(2*DEFAULT_ALPHAS-1) + 
                           (DEFAULT_ALPHAS - 1) * np.exp(-label_eps*DEFAULT_ALPHAS) /(2*DEFAULT_ALPHAS-1)) / DEFAULT_ALPHAS

mask_feat = args.feat
aux_lambda = args.aux

num_epochs = args.ep
learning_rate = args.lr
warm_learning_rate = args.lrw

batch_size_train = args.bz
batch_size_train_warm = args.bzw
batch_size_train_aux = args.bza
MAX_PHYSICAL_BATCH_SIZE = 32


clip_coeff = 1
private = True

warmup_epochs = args.epw
dataset = args.dataset


run_name = "{}_eps{}_delta{}_lbeps{}_lbs{}_feat{}_aux{}_bz{}_bzw{}_bza{}_lr{}_lrw{}_ep{}_epw{}_decay{}".format(dataset,require_eps, require_delta, label_eps,label_scale,mask_feat,aux_lambda,batch_size_train,batch_size_train_warm,batch_size_train_aux,learning_rate,warm_learning_rate,num_epochs,warmup_epochs,args.decay)

if args.model is not None:
    run_name += "_" + args.model

wandb.init(project="DP", entity="t1773420638",name=run_name)


#train_loader, aux_train_loader, and test_loader
# Image preprocessing modules
if dataset == "MNIST":
    transform_train = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

    transform_test = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])


    train_dataset = torchvision.datasets.MNIST(root='.',
                                                train=True, 
                                                download=True, transform=transform_train)


    test_dataset = torchvision.datasets.MNIST(root='.',
                                                train=False, 
                                      train_dataset = torchvision.datasets.MNIST(root='.',
                                            train=True, 
                                            download=True, transform=transform_train)          transform=transform_test)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size_train, 
                                            shuffle=True,num_workers = 2)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1000, 
                                            shuffle=False,num_workers = 2)



    mask_eps_data = MaskedMNIST(label_eps,label_scale,mask_feat,transform=transform_train)


    data_len = len(train_dataset)
    model = MNIST_CNN().to(device)
    
elif dataset == "FMNIST":
    transform_train = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
    
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('./fashionmnist_data/', train=True, download=True,
                    transform=transform_train),
    batch_size=batch_size_train)
    #mask_dataset = MaskedFMNIST(label_eps,label_scale,0,transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(mask_dataset,
    # batch_size=batch_size_train)
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('./fashionmnist_data/', train=False, transform=transform_train),
    batch_size=1000, shuffle=True)
    
    mask_eps_data = MaskedFMNIST(label_eps,label_scale,mask_feat,transform=transform_train)
    #mask_eps_data.eps_y = mask_dataset.eps_y

    data_len = len(mask_eps_data)
    model = MNIST_CNN_Tanh().to(device)
    
elif dataset == "MovieLens":
    pass

elif dataset == "Income":
    pass
    
elif dataset == "CIFAR":
    
    transform_train = transforms.Compose([
    transforms.ColorJitter(),
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧影象隨機裁剪成32*32
    transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每層的歸一化用到的均值和方差
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    train_dataset = torchvision.datasets.CIFAR10(root='.',
                                                train=True, 
                                                download=True, transform=transform_train)

    train_dataset = MakedCIFAR10(label_eps,label_scale,0,transform=transform_train)


    test_dataset = torchvision.datasets.CIFAR10(root='.',
                                                train=False, 
                                                transform=transform_test)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size_train, 
                                            shuffle=True,num_workers = 2)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1000, 
                                            shuffle=False,num_workers = 2)



    mask_eps_data = MakedCIFAR10(label_eps,label_scale,mask_feat,transform=transform_train)
    mask_eps_data.eps_y = train_dataset.eps_y

    data_len = len(train_dataset)
    model = CIFAR_CNN_Tanh().to(device)
    
else:
    raise ValueError("not implemented")

aux_train_loader = torch.utils.data.DataLoader(mask_eps_data,
                                            batch_size=batch_size_train_aux,shuffle=True)
warm_train_loader = torch.utils.data.DataLoader(mask_eps_data,
                                            batch_size=batch_size_train_warm,shuffle=True)


# decide noise
print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(require_eps, require_delta))
sampling_prob=batch_size_train/data_len
steps = int(num_epochs/sampling_prob)
sigma, eps = get_sigma(sampling_prob, steps, require_eps, require_delta, rgp=False)
noise_multiplier = sigma
print('noise scale: ', noise_multiplier, 'privacy guarantee: ', eps)
    


#model.load_state_dict(torch.load("mnist_model_pad0.pth"))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
cross_entropy_noreduce = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
warm_optimizer = torch.optim.SGD(model.parameters(), lr=warm_learning_rate, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Train the model
total_step = data_len // batch_size_train + int((data_len % batch_size_train)>0)
warm_total_step = data_len // batch_size_train_warm + int((data_len % batch_size_train_warm )>0)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
warm_model_name = "warm_{}_lbeps{}_lbs{}_bz{}_lr{}_ep{}_feat{}".format(dataset,label_eps,label_scale,batch_size_train_warm,warm_learning_rate,warmup_epochs,mask_feat)
print(warm_model_name)
        
# Warm up training on public dataset
if os.path.exists("public_pre_train/" + warm_model_name + ".pkl"):
    model.load_state_dict(torch.load("public_pre_train/" + warm_model_name + ".pkl"))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = correct / total
        print('Warm-up : Accuracy of the model on the test images: {} %'.format(100 * test_acc))
        wandb.log({"test acc": test_acc,"warm":1})
else:
    for epoch in range(warmup_epochs):
        model.train()
        for i, (data,target) in enumerate(warm_train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()  
            now_loss = loss.item()
            warm_optimizer.step()
            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, warmup_epochs, i+1, warm_total_step, now_loss))       
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc = correct / total
            print('Warm-up : Accuracy of the model on the test images: {} %'.format(100 * test_acc))
            wandb.log({"test acc": test_acc,"warm":1})
        
    torch.save(model.state_dict(),"public_pre_train/" + warm_model_name + ".pkl")
        
# private training
curr_lr = learning_rate
best_acc = 0
non_improve_step = 0

model.train()

dp_model = GradSampleModule(model)
dp_data_loader = DPDataLoader.from_data_loader(train_loader, distributed=False)
sample_rate = 1 / len(dp_data_loader)
expected_batch_size = int(len(dp_data_loader.dataset) * sample_rate)
dp_optimizer = HybridDPOptimizer(
    optimizer=optimizer,
    noise_multiplier=noise_multiplier,
    max_grad_norm=clip_coeff,
    expected_batch_size=batch_size_train,
)

accountant = RDPAccountant()
dp_optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=sample_rate))


    
aux_data_generator = iter(aux_train_loader)
    
for epoch in range(num_epochs):
    model.train()
    
    with BatchMemoryManager(
        data_loader=dp_data_loader, 
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
        optimizer=dp_optimizer
    ) as memory_safe_data_loader:
    
        for i, (data,target) in enumerate(memory_safe_data_loader):
            dp_optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            outputs = dp_model(data)
            # Backward and optimize
            loss = criterion(outputs, target)
            loss.backward()
            now_loss = loss.item()
            
            done_private = dp_optimizer.pre_step()

            dp_optimizer.zero_grad_sample()

            if done_private:
                aux_loss = 0
                for ii in range(batch_size_train_aux // MAX_PHYSICAL_BATCH_SIZE):
                    try:
                        aux_data, aux_target = next(aux_data_generator)
                    except:
                        aux_data_generator = iter(aux_train_loader)
                        aux_data, aux_target = next(aux_data_generator)
                    aux_data = aux_data.to(device)
                    aux_target = aux_target.to(device)
                    aux_outputs = model(aux_data)
                    aux_loss = - aux_lambda * torch.mean(torch.log_softmax(aux_outputs,1) * aux_target) / (batch_size_train_aux // MAX_PHYSICAL_BATCH_SIZE)
                    aux_loss.backward()
                    dp_optimizer.zero_grad_sample()
                    
                dp_optimizer.step()
            
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = correct / total
        eps_spent, rdp = accountant.get_epsilon(require_delta)
        label_eps_spent, _, _ = get_privacy_spent(DEFAULT_ALPHAS, rdp + label_laplace_rdp, target_delta=require_delta)
        label_eps_spent = min([label_eps_spent, eps_spent + label_eps])
        print(epoch, 'Accuracy of the model on the test images: {} %, curr_lr:{}, curr_best :{}, eps_spent:{}, label_eps_spent:{}'.format(100 * test_acc, curr_lr,best_acc,eps_spent, label_eps_spent))
        wandb.log({"test acc": test_acc,"warm":0,"curr_lr":curr_lr,"eps_spent":eps_spent,"label_eps_spent":label_eps_spent})
    
    if test_acc > best_acc:
        best_acc = test_acc
    #     non_improve_step = 0
    # else:
    #     # if epoch < 60:
    #     #     non_improve_step += 1
    #     #     if non_improve_step >= 5:
    #     #         if curr_lr < 0.00001:
    #     #             break
    #     #         else:
    #     #             curr_lr /= 2
    #     #             update_lr(optimizer, curr_lr)
    #     #             non_improve_step = 0
    if epoch % args.decay == 0:
        curr_lr = curr_lr/2
        update_lr(optimizer, curr_lr)
      
