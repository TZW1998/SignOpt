import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from models import cutout
from torch import autocast 
from torch.cuda.amp import GradScaler 
import torch
import time
from models import *
import os
from vit_pytorch import ViT, SimpleViT

class ClsCIFAR10:
    def __init__(self, batch_size_train = 128,
                           batch_size_test = 1024,
                           cutout_size = 16,
                           num_workers = 4,
                           label_smoothing = 0.1,
                           physical_batchsize = 1024,
                           precison = "amp_half") -> None:
        
        self.batch_size_train = batch_size_train
        self.physical_batchsize = physical_batchsize

        transform_train = transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧影象隨機裁剪成32*32
        transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
        cutout(cutout_size,
            1,
            False),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每層的歸一化用到的均值和方差
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        data_path = "/" + os.path.join(*__file__.split("/")[:-1], 'data')

        train_dataset = torchvision.datasets.CIFAR10(data_path,
                                            train=True, 
                                             transform=transform_train, download=True)


        test_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                            train=False, 
                                            transform=transform_test)


        self.train_loader = DataLoader(train_dataset,
                                        batch_size= self.batch_size_train, 
                                        shuffle=True,num_workers = num_workers)

        self.train_infer_loader = DataLoader(torchvision.datasets.CIFAR10(root=data_path,
                                            train=True, 
                                            download=True, transform=transform_test),
                                        batch_size=batch_size_test, 
                                        shuffle=False,num_workers = num_workers)

        self.test_loader = DataLoader(test_dataset,
                                        batch_size=batch_size_test, 
                                        shuffle=False,num_workers = num_workers)

        self.criterion = nn.CrossEntropyLoss(label_smoothing = label_smoothing)

        self.train_data_len = len(train_dataset)

        self.precison = precison

        if self.precison == "amp_half":
            self.scaler = GradScaler()

        self.start_time = time.time()

    def get_train_loader(self) -> DataLoader:
        return self.train_loader

    def get_train_data_len(self) -> int:
        return self.train_data_len 

    def loss_and_step(self, model, optimizer, batch_data, device) -> float:
        images, target = batch_data

        batch_data_len = len(images)

        if self.physical_batchsize < batch_data_len:

            if (batch_data_len % self.physical_batchsize) == 0:
                num_steps = (batch_data_len // self.physical_batchsize)  
            else: 
                num_steps = (batch_data_len // self.physical_batchsize)   + 1
     
            for j in range(num_steps):
                if j == (num_steps - 1):
                    input_var = images[(j*self.physical_batchsize):].to(device)
                    target_var = target[(j*self.physical_batchsize):].to(device)
                else:
                    input_var = images[(j*self.physical_batchsize):((j+1)*self.physical_batchsize)].to(device)
                    target_var = target[(j*self.physical_batchsize):((j+1)*self.physical_batchsize)].to(device)
                
                input_var = input_var.half() if self.precison == "half" else input_var

                if self.precison == "amp_half":
                    with autocast(device_type='cuda', dtype=torch.float16):
                        # compute output
                        output = model(input_var)
                        loss = self.criterion(output, target_var) * len(input_var) / self.batch_size_train
                    self.scaler.scale(loss).backward()

                else:
                    output = model(input_var)
                
                    loss = self.criterion(output, target_var) * len(input_var) / self.batch_size_train
                    loss.backward()

        else:
            input_var = images.to(device)
            target_var = target.to(device)

            input_var = input_var.half() if self.precison == "half" else input_var
            
            if self.precison == "amp_half":
                with autocast(device_type='cuda', dtype=torch.float16):
                    # compute output
                    output = model(input_var)
                    loss = self.criterion(output, target_var)
                self.scaler.scale(loss).backward()
            else:
            # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)
                loss.backward()

        if self.precison == "amp_half":
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()  

        return loss.item()

    def eval_model_with_log(self,model, device) -> None:
        with torch.no_grad():
        # train acc/loss
            model.eval()
            train_loss, train_total, train_correct = 0.0, 0.0, 0.0
            for batch_idx, (images1, labels1) in enumerate(self.train_infer_loader):
                images, labels = images1.to(device), labels1.to(device)

                images = images.half() if self.precison == "half" else images
                # Inference
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                train_loss += batch_loss.item() * len(labels)
                train_total += len(labels)

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                train_correct += torch.sum(torch.eq(pred_labels, labels)).item()
            train_loss /= train_total
            train_correct /= train_total

            # test acc/loss
            test_loss, test_total, test_correct = 0.0, 0.0, 0.0
            for batch_idx, (images1, labels1) in enumerate(self.test_loader):
                images, labels = images1.to(device), labels1.to(device)
                images = images.half() if self.precison == "half" else images
                # Inference
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                test_loss += batch_loss.item() * len(labels)
                test_total += len(labels)

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                test_correct += torch.sum(torch.eq(pred_labels, labels)).item()
            test_loss /= test_total
            test_correct /= test_total

        dt = {
            'train_loss': train_loss,
            'train_acc': train_correct,
            'test_loss': test_loss,
            'test_acc': test_correct,
            'time': time.time() - self.start_time}

        model.train() # switch back to train mode
        return dt
    
    def get_model(self, model_name):
        if model_name == "ViT-S":
            model = SimpleViT(
                    image_size = 32,
                    patch_size = 4,
                    num_classes = 10,
                    dim = 256,
                    depth = 6,
                    heads = 12,
                    mlp_dim = 1024
                )
        elif model_name == "ViT-B":
            model = SimpleViT(
                    image_size = 32,
                    patch_size = 4,
                    num_classes = 10,
                    dim = 512,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 1024
                )
        elif model_name == "ViT-L":
            model = SimpleViT(
                    image_size = 32,
                    patch_size = 4,
                    num_classes = 10,
                    dim = 1024,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048
                )
        elif model_name == "ViT-H":
            model = SimpleViT(
                    image_size = 32,
                    patch_size = 4,
                    num_classes = 10,
                    dim = 1024,
                    depth = 9,
                    heads = 24,
                    mlp_dim = 2048
                )
        elif model_name == "ConvNeXt-S":
            model = ConvNeXt(10,
                 channel_list = [64, 128, 128, 128],
                 num_blocks_list = [1, 1, 1, 1],
                 kernel_size=7, patch_size=1,
                 res_p_drop=0.)
        elif model_name == "ConvNeXt-B":
            model = ConvNeXt(10,
                 channel_list = [64, 128, 256, 512],
                 num_blocks_list = [2, 2, 2, 2],
                 kernel_size=7, patch_size=1,
                 res_p_drop=0.)
        elif model_name == "ConvNeXt-L":
            model = ConvNeXt(10,
                 channel_list = [256, 512, 512, 1024],
                 num_blocks_list = [2, 2, 2, 2],
                 kernel_size=7, patch_size=1,
                 res_p_drop=0.)
        elif model_name == "ConvNeXt-H":
           model = ConvNeXt(10,
                 channel_list = [256, 512, 1024, 1024, 1024],
                 num_blocks_list = [2, 2, 2, 2, 2],
                 kernel_size=7, patch_size=1,
                 res_p_drop=0.)
        else:
            raise ValueError("model name not found")
        
        return model
