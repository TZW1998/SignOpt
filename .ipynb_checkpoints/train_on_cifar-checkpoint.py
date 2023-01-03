import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy import io as spio

from scipy.stats import norm
import wandb
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_epoch', default = 50, type=int)
parser.add_argument('--lr_client', default = 0.1, type=float)
parser.add_argument('--lr_server', default = 1, type=float)
parser.add_argument('--local_bach_size', default = 32, type=int)
parser.add_argument('--noise_multiplier_sigma', default = 0.0, type=float)
parser.add_argument('--use_sign', action="store_true")
parser.add_argument('--wandb', action="store_true")
parser.add_argument('--momentum', default = 0, type=float)
parser.add_argument('--data_distribution', default = 4, type=int, choices=range(1, 4))
parser.add_argument('--alpha', default = 1, type=float)
parser.add_argument('--plot_interval', default = 100, type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--mark', type=int)

args = parser.parse_args()
# define paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# FL
total_clients = 100
clients_number_per_round = args.clients_number_per_round # total: 1000
local_update_step_number = args.local_update_step_number # FedSGD
num_rounds = args.num_rounds

# local model updating
lr_client = args.lr_client # learning rate of client
lr_server = args.lr_server # learning rate of server
local_bach_size = args.local_bach_size
momentum = args.momentum

# DP noise
noise_multiplier_sigma = args.noise_multiplier_sigma
use_sign = args.use_sign

# file name
if args.data_distribution == 4:
    run_name = "new_clients_{}_lsteps_{}_rounds_{}_lrc_{}_lrs_{}_bz_{}_noise_{}_sign_{}_momentum_{}_data_{}_alpha_{}_inter_{}_seed_{}_mark_{}".format(clients_number_per_round, local_update_step_number,
                                                                                                                        num_rounds, lr_client, lr_server,
                                                                                                                        local_bach_size, noise_multiplier_sigma,
                                                                                                                        use_sign, momentum,
                                                                                                                        args.data_distribution, args.alpha,
                                                                                                                        args.plot_interval, args.seed,args.mark)
else:
    run_name = "brand_clients_{}_lsteps_{}_rounds_{}_lrc_{}_lrs_{}_bz_{}_noise_{}_sign_{}_momentum_{}_data_{}_inter_{}_seed_{}".format(clients_number_per_round, local_update_step_number,
                                                                                                                        num_rounds, lr_client, lr_server,
                                                                                                                        local_bach_size, noise_multiplier_sigma,
                                                                                                                        use_sign,momentum,
                                                                                                                        args.data_distribution,args.plot_interval,args.seed)
print(run_name)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
if args.wandb:
    wandb.init(project="DP-FL", entity="t1773420638",name=run_name)

# ============= Neural network : ResNet20 + GN ============= #
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class NeuralNet(nn.Module):
    def __init__(self, num_classes=10):
        super(NeuralNet, self).__init__()
        self.in_planes = 16

        # initial
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()

        # block 1
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu12 = nn.ReLU()
        self.shortcut12 = nn.Sequential()

        self.conv13 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu14 = nn.ReLU()
        self.shortcut14 = nn.Sequential()

        self.conv15 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.GroupNorm(4, 16)  # nn.BatchNorm2d(16)
        self.relu16 = nn.ReLU()
        self.shortcut16 = nn.Sequential()

        # block 2
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn21 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu21 = nn.ReLU()
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu22 = nn.ReLU()
        self.shortcut22 = LambdaLayer(lambda x:
                                      F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 32 // 4, 32 // 4), "constant", 0))

        self.conv23 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu23 = nn.ReLU()
        self.conv24 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu24 = nn.ReLU()
        self.shortcut24 = nn.Sequential()

        self.conv25 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu25 = nn.ReLU()
        self.conv26 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.GroupNorm(8, 32)  # nn.BatchNorm2d(32)
        self.relu26 = nn.ReLU()
        self.shortcut26 = nn.Sequential()

        # block 3
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn31 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu31 = nn.ReLU()
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu32 = nn.ReLU()
        self.shortcut32 = LambdaLayer(lambda x:
                                      F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 64 // 4, 64 // 4), "constant", 0))

        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu33 = nn.ReLU()
        self.conv34 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn34 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu34 = nn.ReLU()
        self.shortcut34 = nn.Sequential()

        self.conv35 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn35 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu35 = nn.ReLU()
        self.conv36 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn36 = nn.GroupNorm(16, 64)  # nn.BatchNorm2d(64)
        self.relu36 = nn.ReLU()
        self.shortcut36 = nn.Sequential()

        # final
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        # initial
        out0 = self.relu0(self.bn0(self.conv0(x)))

        # block 1
        out = self.relu11(self.bn11(self.conv11(out0)))
        out = self.bn12(self.conv12(out))
        out += self.shortcut12(out0)
        out12 = self.relu12(out)

        out = self.relu13(self.bn13(self.conv13(out12)))
        out = self.bn14(self.conv14(out))
        out += self.shortcut14(out12)
        out14 = self.relu14(out)

        out = self.relu15(self.bn15(self.conv15(out14)))
        out = self.bn16(self.conv16(out))
        out += self.shortcut16(out14)
        out16 = self.relu16(out)

        # block 2
        out = self.relu21(self.bn21(self.conv21(out16)))
        out = self.bn22(self.conv22(out))
        out += self.shortcut22(out16)
        out22 = self.relu22(out)

        out = self.relu23(self.bn23(self.conv23(out22)))
        out = self.bn24(self.conv24(out))
        out += self.shortcut24(out22)
        out24 = self.relu24(out)

        out = self.relu25(self.bn25(self.conv25(out24)))
        out = self.bn26(self.conv26(out))
        out += self.shortcut26(out24)
        out26 = self.relu26(out)

        # block 3
        out = self.relu31(self.bn31(self.conv31(out26)))
        out = self.bn32(self.conv32(out))
        out += self.shortcut32(out26)
        out32 = self.relu32(out)

        out = self.relu33(self.bn33(self.conv33(out32)))
        out = self.bn34(self.conv34(out))
        out += self.shortcut34(out32)
        out34 = self.relu34(out)

        out = self.relu35(self.bn35(self.conv35(out34)))
        out = self.bn36(self.conv36(out))
        out += self.shortcut36(out34)
        out36 = self.relu36(out)

        # final
        out = F.avg_pool2d(out36, out36.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

# ============= Database distribution (iid, non-iid) ============= #
def CIFAR10_iid(dataset, num_users, weight_set):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        num_items = int(np.round(len(dataset) * weight_set[i]))
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def CIFAR10_noniid_1bit(dataset, num_users, weight_set):
    # 60,000 training imgs -->  2000 imgs/shard X 30 shards
    num_shards, num_imgs = len(dataset), 1
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign shards/client
    start_index = 0
    for i in range(num_users):
        rand_set = set(np.arange(start_index, start_index + int(np.round(num_shards * weight_set[i]))))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        start_index = start_index + int(np.round(num_shards * weight_set[i]))

    return dict_users

def CIFAR10_noniid_2bits(dataset, num_users):
    idx_shard = [i for i in range(len(dataset))]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign shards/client
    classes_each_device = 2
    data_index = np.random.choice(num_users*classes_each_device,num_users*classes_each_device,replace=False)
    idx = 0
    for i in range(int(num_users)):
        for j in range(classes_each_device):
            rand_set = np.arange(data_index[idx]*int(len(dataset)/num_users/classes_each_device), (data_index[idx]+1)*int(len(dataset)/num_users/classes_each_device))
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand_set]), axis=0)
            idx += 1
    return dict_users

def CIFAR10_noniid_diff_label_prob(dataset, num_users, weight_per_label_set):
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    classes_each_device = 10
    start_index = [int(len(dataset) / classes_each_device * j) for j in range(classes_each_device)]

    for j in range(classes_each_device):
        for i in range(int(num_users) - 1):
            rand_set = np.arange(start_index[j], start_index[j] + int(
                np.round(len(dataset) / classes_each_device * weight_per_label_set[i][j])))
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand_set]), axis=0)
            start_index[j] = start_index[j] + int(
                np.round(len(dataset) / classes_each_device * weight_per_label_set[i][j]))
        i = int(num_users) - 1
        rand_set = np.arange(start_index[j], int(len(dataset) / classes_each_device * (j + 1)))
        dict_users[i] = np.concatenate((dict_users[i], idxs[rand_set]), axis=0)

    return dict_users

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# ============= Seed Setting ============= #
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# training process
if os.path.exists("results_CIFAR10_FedSign_2bits/"+run_name+".txt"):
    print("Run already exists:")
else:
    print("start training:")

    gpu_id = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if args.seed is not None:
        setup_seed(args.seed)

    # ============= Database CIFAR-10 ============= #
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    train_all_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1280, shuffle=False, num_workers=0, pin_memory=True)
    test_all_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1280, shuffle=False, num_workers=0, pin_memory=True)

    # iid / non-iid data distribution
    if args.data_distribution == 1: # iid
        node_groups = CIFAR10_iid(train_dataset, total_clients, np.ones(total_clients) / total_clients)
    elif args.data_distribution == 2: # non-iid
        node_groups = CIFAR10_noniid_1bit(train_dataset, total_clients, np.ones(total_clients) / total_clients)
    elif args.data_distribution == 3: # non-iid
        node_groups = CIFAR10_noniid_2bits(train_dataset, total_clients)
    else: # non-iid: Dirichlet distribution (each client has non-uniform distribution)
        weight_per_label_set = np.random.dirichlet(np.ones(10) * args.alpha, total_clients)
        weight_per_label_set /= np.sum(weight_per_label_set, 0)
        node_groups = CIFAR10_noniid_diff_label_prob(train_dataset, total_clients, weight_per_label_set)

    # ============= Training ============= #
    global_model = NeuralNet().to(device)
    model_diff_average_momentum_dict = {n:torch.zeros_like(p) for n,p in global_model.named_parameters()}
    with open("results_CIFAR10_FedSign_2bits/"+run_name+".txt","w") as f:
        for round in range(num_rounds):
            choosed_clients = np.random.choice(total_clients,clients_number_per_round,replace=False).astype(int)
            model_diff_average_dict = {n:torch.zeros_like(p) for n,p in global_model.named_parameters()}
            global_state_dict = copy.deepcopy(global_model.state_dict())

            for client in choosed_clients:
                client_writer_indexes = list(node_groups[client])
                train_dataset_each_client = DatasetSplit(train_dataset, client_writer_indexes)
                    
                # mini-batch samples
                client_train_loader = torch.utils.data.DataLoader(train_dataset_each_client,
                                                                  batch_size=local_bach_size, shuffle=False,
                                                                  sampler=torch.utils.data.sampler.RandomSampler(
                                                                      range(len(train_dataset_each_client)),
                                                                      replacement=True,
                                                                      num_samples=local_bach_size * local_update_step_number),
                                                                  num_workers=0, pin_memory=True)

                local_model = NeuralNet().to(device)
                local_model.load_state_dict(global_state_dict)   
                local_optimizer = torch.optim.SGD(local_model.parameters(),lr_client,momentum = 0,weight_decay = 0)
                # local update
                local_step = 0
                while local_step < local_update_step_number:
                    local_model.train()
                    for i, (images, target) in enumerate(client_train_loader):
                        local_step += 1

                        input_var = images.to(device)
                        target_var = target.to(device)
                        
                        # compute output
                        output = local_model(input_var)
                        loss = criterion(output, target_var)

                        # compute gradient and do SGD step
                        local_optimizer.zero_grad()
                        loss.backward()
                        local_optimizer.step()
                        
                local_model_state_dict = local_model.state_dict() 
                local_model_clip = torch.sqrt(sum(torch.norm(local_model_state_dict[n] - p.data)**2 for n,p in global_model.named_parameters()))
                with torch.no_grad():
                    for n,p in global_model.named_parameters():    
                        temp_diff = local_model_state_dict[n] - p.data
                        if noise_multiplier_sigma > 0:
                            temp_diff += torch.Tensor(temp_diff.size()).normal_(0, noise_multiplier_sigma).cuda()
                        if use_sign:
                            model_diff_average_dict[n] += torch.sign(temp_diff)
                        else:
                            model_diff_average_dict[n] += temp_diff
            # average
            for n,p in global_model.named_parameters():    
                model_diff_average_dict[n] /= clients_number_per_round
            
            # update momentum
            for n,p in global_model.named_parameters():
                model_diff_average_momentum_dict[n] = momentum * model_diff_average_momentum_dict[n] + model_diff_average_dict[n]
            
            # global model step
            with torch.no_grad():
                for n,p in global_model.named_parameters():
                    p.data += lr_server * model_diff_average_momentum_dict[n]

#             if (round+1) % 1500 == 0:
#                 lr_client /= 2
                
            
            #check
            if (round + 1) % args.plot_interval == 0:
                with torch.no_grad():
                    # train acc/loss
                    train_loss, train_total, train_correct = 0.0, 0.0, 0.0
                    for batch_idx, (images1, labels1) in enumerate(train_all_loader):
                        images, labels = images1.to(device), labels1.to(device)

                        # Inference
                        outputs = global_model(images)
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
                    for batch_idx, (images1, labels1) in enumerate(test_all_loader):
                        images, labels = images1.to(device), labels1.to(device)

                        # Inference
                        outputs = global_model(images)
                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item() * len(labels)
                        test_total += len(labels)

                        _, pred_labels = torch.max(outputs, 1)
                        pred_labels = pred_labels.view(-1)
                        test_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                    test_loss /= test_total
                    test_correct /= test_total

                print("round:", round, "train loss:", train_loss, "train acc", train_correct, "test loss:", test_loss, "test acc", test_correct)
                if args.wandb:
                    wandb.log({"train loss": train_loss,
                               "train acc": train_correct,
                               "test loss": test_loss,
                               "test acc": test_correct})
                stats = "{},{},{},{},{}\n".format(round,train_loss,train_correct,test_loss,test_correct)
                f.write(stats)
                f.flush()
            
    with open("total_results.txt","a") as f:
        f.write(run_name+"->"+str(test_correct)+"\n")
