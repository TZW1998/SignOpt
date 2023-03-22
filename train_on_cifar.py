import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tensorboardX import SummaryWriter
from torch import autocast 
from torch.cuda.amp import GradScaler 

import numpy as np
import random
import copy

import argparse
import os, shutil
import time

from optimizers import *
from models import *


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--gpu", default = "0")
parser.add_argument("--run_name")
parser.add_argument("--rerun", type = bool)
parser.add_argument("--rep", type = int)

parser.add_argument('--num_epoch', default = 80, type=int)
parser.add_argument('--lr', default = 0.001, type=float)
parser.add_argument('--lr_schedule', default = "none", type=str,
                 choices=["plateau", "cosine", "none", "step", "onecycle"])
parser.add_argument('--MulStep', default = [60,80], type=int, nargs='+')
parser.add_argument('--batchsize', default = 4096, type=int)
parser.add_argument('--optimizer', default = "SignSGD", type=str, choices=["SGD", "Adam", "SignSGD", "Lamb"])
parser.add_argument('--tensorboard', action="store_true")
parser.add_argument('--momentum', default = 0.9, type=float)
parser.add_argument('--weight_decay', default = 0, type=float)
parser.add_argument('--beta1', default = 0.9, type=float)
parser.add_argument('--beta2', default = 0.999, type=float)
parser.add_argument('--tol', default = 10, type=int)
parser.add_argument('--plot_interval', default = 1000, type=int)
parser.add_argument('--model', default = "fastnet", type=str, choices=["resnet18", "resnet18_pretrained", "wide_resnet50_2", "resnet20","fastnet","fastnet2","resnet56P"])
parser.add_argument('--precison', type=str, default="amp_half", choices=["single", "half", "amp_half"])
parser.add_argument('--seed', type=int)
parser.add_argument('--test_batchsize', type=int, default = 1024)
parser.add_argument('--physical_batchsize', type=int, default = 1024)
parser.add_argument('--label_smooth', type=float, default = 0.1)
parser.add_argument('--cutout', type=int, default = 16)
parser.add_argument('--noise_scale', type=float, default = 0)
parser.add_argument('--max_lr', type=float, default = 0.4)
parser.add_argument('--SignTyp', type=str, default = "standard", choices=["standard","EF","layerEF","1norm","layer1norm"])
parser.add_argument('--quant_init', action="store_true")
parser.add_argument('--linf_bound', type=float, default = -1)

args = parser.parse_args()
batch_size_train = args.batchsize
lr = args.lr

if args.optimizer == "SGD":
    optimizer_name = "(SGD,lr:{},mt:{}".format(args.lr,args.momentum)
elif args.optimizer == "Adam":
    optimizer_name = "(Adam,lr:{},b1:{},b2:{})".format(args.lr,args.beta1,args.beta2)
elif args.optimizer == "SignSGD":
    optimizer_name = "(SignSGD,lr:{},,mt:{},ns:{},st:{})".format(args.lr,args.momentum,args.noise_scale,args.SignTyp)
elif args.optimizer == "Lamb":
    optimizer_name = "(Lamb,lr:{},,b1:{},b2:{})".format(args.lr,args.beta1,args.beta2)

if args.lr_schedule == "plateau":
    lr_schedule_name = "(plateau,tol:{})".format(args.tol)
elif args.lr_schedule == "cosine":
    lr_schedule_name = "(cosine,tol:{})".format(args.tol)
elif args.lr_schedule == "none":
    lr_schedule_name = "none"
elif args.lr_schedule == "step":
    lr_schedule_name = "(step,ms:{})".format(args.MulStep)
elif args.lr_schedule == "onecycle":
    lr_schedule_name = "(onecycle,mlr:{})".format(args.max_lr)

run_tags = "{}_{}_{}_{}_qi:{}_linf:{}_opt:{}_lrs:{}_bz:{}_wd:{}_lbs:{}_ct:{}".format(args.run_name,
                                                            args.rep,
                                                            args.model, 
                                                            args.precison,
                                                            args.quant_init,
                                                            args.linf_bound,
                                                            optimizer_name,
                                                            lr_schedule_name,
                                                            args.batchsize,
                                                            args.weight_decay,
                                                            args.label_smooth,
                                                            args.cutout)



def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if os.path.exists("code/runs/{}".format(run_tags)):
    clean_dir("code/runs/{}".format(run_tags))
writer = SummaryWriter(logdir='code/runs/{}'.format(run_tags))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# training process
if os.path.exists("code/logs/"+run_tags+".txt") and False: #(not args.rerun):
    print("Run already exists:")
else:
    print("start training:")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if args.seed is not None:
        setup_seed(args.seed)

    # define paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transform_train = transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧影象隨機裁剪成32*32
        transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
        cutout(args.cutout,
               1,
               False),
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


    test_dataset = torchvision.datasets.CIFAR10(root='.',
                                                train=False, 
                                                transform=transform_test)


    train_loader = DataLoader(train_dataset,
                                            batch_size= batch_size_train, 
                                            shuffle=True,num_workers = 4)

    train_infer_loader = DataLoader(torchvision.datasets.CIFAR10(root='.',
                                                train=True, 
                                                download=True, transform=transform_test),
                                            batch_size=args.test_batchsize, 
                                            shuffle=False,num_workers = 4)

    test_loader = DataLoader(test_dataset,
                                            batch_size=args.test_batchsize, 
                                            shuffle=False,num_workers = 4)
    
    if args.model == "resnet56P":
        model = resnet56P().to(device)
    elif args.model == "fastnet":
        model = fastnet().to(device)
    elif args.model == "fastnet2":
        model = fastnet2().to(device)
    elif args.model == "resnet20":
        model = resnet20().to(device)
    elif "pretrained" in args.model:
        model = torch.hub.load('pytorch/vision:v0.10.0', args.model.strip("_pretrained"), pretrained=True).to(device)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=False).to(device)

    if args.precison == "half":
        model = model.half()
    elif args.precison == "amp_half":
        scaler = GradScaler()

    if args.quant_init:
        for p in model.parameters():
            p.data.div_(lr,rounding_mode="floor").mul_(lr)

    torch.save(model,"init_model.pt")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        if args.weight_decay > 1e-7:
            optimizer = torch.optim.AdamW(model.parameters(), lr = lr, betas = (args.beta1, args.beta2), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (args.beta1, args.beta2)) 
    elif args.optimizer == "SignSGD":
        optimizer = SignSGD(model.parameters(), lr = lr, momentum = args.momentum, weight_decay=args.weight_decay, noise_scale=args.noise_scale, sign_type = args.SignTyp)
    elif args.optimizer == "Lamb":
        optimizer = Lamb(model.parameters(), lr = lr, betas = (args.beta1, args.beta2))
        if args.weight_decay > 1e-7:
            Warning("Lamb does not support weight decay")
    else:
        raise NotImplementedError
    
    if args.lr_schedule == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', 
        patience = args.tol * len(train_dataset) // (args.plot_interval * batch_size_train), 
        min_lr = 1e-4)
    elif args.lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer,
         T_max = args.tol * len(train_dataset) // (args.plot_interval * batch_size_train),
         eta_min=1e-4)
    elif args.lr_schedule == "none":
        scheduler = None
    elif args.lr_schedule == "step":
        scheduler = MultiStepLR(optimizer, milestones=args.MulStep, gamma=0.1)
    elif args.lr_schedule == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_loader), epochs=args.num_epoch)
    else:
        raise NotImplementedError

    steps = 0
    start_time = time.time()

    if args.linf_bound >= 2:
        lower_range = - 2. ** (args.linf_bound - 1) 
        upper_range = - lower_range - 1
        lower_range *= lr
        upper_range *= lr
        print("lower_range: ", lower_range, "upper_range: ", upper_range)
    
    # ============= Training ============= #
    with open("code/logs/"+run_tags+".txt","w") as f:
        for ep in range(args.num_epoch):
            for i, (images, target) in enumerate(train_loader):
                steps += 1
                optimizer.zero_grad()
                if args.physical_batchsize < batch_size_train:

                    if (batch_size_train % args.physical_batchsize) == 0:
                        num_steps = (batch_size_train // args.physical_batchsize)  
                    else: 
                        num_steps = (batch_size_train // args.physical_batchsize)   + 1
                        
                    for j in range(num_steps):
                        if j == num_steps - 1:
                            input_var = images[j*args.physical_batchsize:].to(device)
                            target_var = target[j*args.physical_batchsize:].to(device)
                        else:
                            input_var = images[(j*args.physical_batchsize):((j+1)*args.physical_batchsize)].to(device)
                            target_var = target[(j*args.physical_batchsize):((j+1)*args.physical_batchsize)].to(device)

                        input_var = input_var.half() if args.precison == "half" else input_var

                        if args.precison == "amp_half":
                            with autocast(device_type='cuda', dtype=torch.float16):
                                # compute output
                                output = model(input_var)
                                loss = criterion(output, target_var) * len(input_var) / batch_size_train
                            scaler.scale(loss).backward()

                        else:
                            output = model(input_var)
                        
                            loss = criterion(output, target_var) * len(input_var) / batch_size_train
                            loss.backward()
    
                else:
                    input_var = images.to(device).half() if args.precison == "half" else images.to(device)
                    target_var = target.to(device)
                    
                    if args.precison == "amp_half":
                        with autocast(device_type='cuda', dtype=torch.float16):
                            # compute output
                            output = model(input_var)
                            loss = criterion(output, target_var)
                        scaler.scale(loss).backward()
                    else:
                    # compute output
                        output = model(input_var)
                        loss = criterion(output, target_var)
                        loss.backward()
                # compute gradient and do SGD step
                

                if args.precison == "amp_half":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()  

                if args.lr_schedule == "onecycle":
                    scheduler.step()

                # projection to linf ball
                if args.linf_bound >= 2:
                    for param in model.parameters():
                        param.data = param.data.clamp(lower_range, upper_range)
            
                #check
                if ((steps + 1) % args.plot_interval == 0) or ((ep == args.num_epoch - 1) and (i == len(train_loader) - 1)):
                    with torch.no_grad():
                        # train acc/loss
                        model.eval()
                        train_loss, train_total, train_correct = 0.0, 0.0, 0.0
                        for batch_idx, (images1, labels1) in enumerate(train_infer_loader):
                            images, labels = images1.to(device), labels1.to(device)

                            images = images.half() if args.precison == "half" else images
                            # Inference
                            outputs = model(images)
                            batch_loss = criterion(outputs, labels)
                            train_loss += batch_loss.item() * len(labels)
                            train_total += len(labels)

                            _, pred_labels = torch.max(outputs, 1)
                            pred_labels = pred_labels.view(-1)
                            train_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                        train_loss /= train_total
                        train_correct /= train_total

                        # test acc/loss
                        test_loss, test_total, test_correct = 0.0, 0.0, 0.0
                        for batch_idx, (images1, labels1) in enumerate(test_loader):
                            images, labels = images1.to(device), labels1.to(device)
                            images = images.half() if args.precison == "half" else images
                            # Inference
                            outputs = model(images)
                            batch_loss = criterion(outputs, labels)
                            test_loss += batch_loss.item() * len(labels)
                            test_total += len(labels)

                            _, pred_labels = torch.max(outputs, 1)
                            pred_labels = pred_labels.view(-1)
                            test_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                        test_loss /= test_total
                        test_correct /= test_total

                    if scheduler is None:
                        lr_now = lr
                    else:
                        lr_now = scheduler.get_last_lr()[0]


                    dt = {
                            'epoch': ep,
                            'train_loss': train_loss,
                            'train_acc': train_correct,
                            'test_loss': test_loss,
                            'test_acc': test_correct,
                            'lr': lr_now,
                            'time': time.time() - start_time}

                    print("epoch:", ep, "steps:",steps, "train loss:", train_loss, "train acc", train_correct, "test loss:", test_loss, "test acc", test_correct, "lr", lr_now, "time", dt["time"])



                    if args.tensorboard or True:
                        
                        for k, v in dt.items():
                            writer.add_scalar("stats/" + k, v, steps)


                    if args.lr_schedule == "plateau":
                        scheduler.step(test_loss)
                    elif args.lr_schedule == "cosine":
                        scheduler.step()
                    else:
                        pass

                    stats = ",".join(["{}".format(v) for v in dt.values()])
                    f.write(stats)
                    f.flush()

                model.train()

            if args.lr_schedule == "step":
                scheduler.step()
writer.close()
torch.save(model,"after_model.pt")
