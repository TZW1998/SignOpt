import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tensorboardX import SummaryWriter
import numpy as np
import random
import copy
import argparse
import os, shutil
import time
from optimizers import *
from models import *
from tasks import *
import json

def load_args():
    parser = argparse.ArgumentParser(description='Process some args.')

    # resume args
    parser.add_argument("--resume", type = str, default = None, description = "to resume from a previous unfinised run")

    # load args from config file
    parser.add_argument("--config", type = str, default = None, description = "to load args from a config file")

    # main experiment args
    parser.add_argument("--gpu", default = "0", description = "gpu id")
    parser.add_argument("--run_name", default = "test", description = "name for this run")
    parser.add_argument("--rerun", type = bool, default = False, description = "whether to rerun this run even though it has been run before")
    parser.add_argument('--seed', type=int, default=42, description = "random seed")

    # task args
    parser.add_argument('--task', default = "diff_MNIST", choices=["cls_CIFAR10","diff_MNIST"], type=str, description = "task to run")
    parser.add_argument('--model', default = "fastnet", type=str, choices=["resnet18", "resnet18_pretrained", "wide_resnet50_2", "resnet20","fastnet","fastnet2","resnet56P", "unet"], description = "model architecture to use, be careful that some models are only for specific tasks")

    # args for global averaging
    parser.add_argument('--num_glb_rounds', default = 100, type=int, description = "number of global averaging rounds")
    parser.add_argument('--num_nodes', default = 8, type=int, description = "number of parelle nodes")
    parser.add_argument('--glb_lr', default = 0.001, type=float, description = "learning rate for global averaging")
    parser.add_argument('--glb_lr_schedule', default = "none", type=str,
                    choices=["plateau", "none", "step"], description = "learning rate schedule for global averaging")
    parser.add_argument('--glb_MulStep', default = [60,80], type=int, nargs='+', description = "milestones for step learning rate schedule, only used when lr_schedule is step")
    parser.add_argument('--glb_tol', default = 10, type=int, description = "tolerance for plateau learning rate schedule, only used when lr_schedule is plateau")
    parser.add_argument('--glb_optimizer', default = "SGD", type=str, choices=["SGD", "Adam", "AdamW", "SignSGD"])
    parser.add_argument('--glb_beta1', default = 0.9, type=float, description = "For all optimizers, this is related to momentum")
    parser.add_argument('--glb_noise_scale', type=float, default = 0, description = "noise scale for the sign-based compression, only used when optimizer is SignSGD")
    parser.add_argument('--glb_beta2', default = 0.99, type=float, description = "For Adam, AdamW, and Lion, this is about the adaptive step, for SignSGD this is related to updated momentum")
    parser.add_argument('--glb_weight_decay', default = 0, type=float, description = "weight decay for global averaging")

    # args for local training
    parser.add_argument('--num_local_steps', default = 1, type=int, description = "number of local training steps")
    parser.add_argument('--batchsize', default = 32, type=int, description = "batch size for local training")
    parser.add_argument('--lr', default = 0.001, type=float, description = "learning rate for local training, for local training, we use a fixed learning rate")
    parser.add_argument('--optimizer', default = "SGD", type=str, choices=["SGD", "Adam", "AdamW", "SignSGD"])
    parser.add_argument('--beta1', default = 0.9, type=float, description = "For all optimizers, this is related to momentum")
    parser.add_argument('--beta2', default = 0.99, type=float, description = "For Adam, AdamW, and SignSGD, this is about the adaptive step, for SignSGD this is related to updated momentum")
    parser.add_argument('--weight_decay', default = 0, type=float, description = "weight decay for local training")

    # args for logging and saving
    parser.add_argument('--log_tool', default = "tensorboard", type=str, choices=["tensorboard", "wandb"], description = "logging tool")
    parser.add_argument('--plot_interval', default = 10, type=int, description = "interval for logging")
    parser.add_argument('--save_interval', default = 20, type=int, description = "interval for saving")

    # args for torch training/testing
    parser.add_argument('--precision', type=str, default="amp_half", choices=["single", "half", "amp_half"])
    parser.add_argument('--test_batchsize', type=int, default = 1024, description = "batch size for testing")
    parser.add_argument('--physical_batchsize', type=int, default = 1024, description = "physical batch size for training, when physical_batchsize > batchsize, we use gradient accumulation")

    args = parser.parse_args()

    # whether to load other args from a config file
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print("load args from config file: " + args.config)
        for key in config:
            setattr(args, key, config[key])
    # resume from a previous run
    elif args.resume is not None: 
        config_path = os.path.join(args.resume, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("resume from a previous run: " + args.resume)
        for key in config:
            if key not in ["resume", "gpu"]: # allow to change gpu for resume
                setattr(args, key, config[key])

    return parser.parse_args()

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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# training process
if os.path.exists("code/logs/"+run_tags+".txt") and (not args.rerun): #and False: :
    print("Run already exists:")
else:
    print("start training:")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if args.seed is not None:
        setup_seed(args.seed)

    # define paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get training task
    if args.task == "cls_CIFAR10":
        task = ClsCIFAR10(batch_size_train = batch_size_train,
                           batch_size_test = args.test_batchsize,
                           cutout_size = args.cutout,
                           num_workers = 4,
                           label_smoothing = args.label_smooth,
                           physical_batchsize = args.physical_batchsize,
                           precison = args.precison)
                    
    elif args.task == "diff_MNIST":
        task = DiffMNIST(batch_size_train = batch_size_train,
                          num_workers = 4,
                          physical_batchsize = args.physical_batchsize,
                          precison = args.precison)
    else:
        raise NotImplementedError

    train_loader = task.get_train_loader()
    
    if args.task == "diff_MNIST":
        model = unet().to(device)
        if args.model != "unet":
            Warning("Model can only be unet for diff_MNIST task")
    elif args.model == "resnet56P":
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
        

    if args.quant_init:
        for p in model.parameters():
            p.data.div_(lr,rounding_mode="floor").mul_(lr)

    # torch.save(model,"init_model.pt")

    # setup optimizer
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
    

    train_data_len = task.get_train_data_len()

    # setup lr scheduler
    if args.lr_schedule == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', 
        patience = args.tol * train_data_len // (args.plot_interval * batch_size_train), 
        min_lr = 1e-4)
    elif args.lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer,
         T_max = args.tol * train_data_len // (args.plot_interval * batch_size_train),
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

    if args.linf_bound >= 2:
        lower_range = - 2. ** (args.linf_bound - 1) 
        upper_range = - lower_range - 1
        lower_range *= lr
        upper_range *= lr
        print("lower_range: ", lower_range, "upper_range: ", upper_range)
    
    # ============= Training ============= #
    with open("code/logs/"+run_tags+".txt","w") as f:
        for ep in range(args.num_epoch):
            for i, batch_data in enumerate(train_loader):
                steps += 1
                optimizer.zero_grad()
                
                loss = task.loss_and_step(model, optimizer, batch_data, device)

                if args.lr_schedule == "onecycle":
                    scheduler.step()

                # projection to linf ball
                if args.linf_bound >= 2:
                    for param in model.parameters():
                        param.data = param.data.clamp(lower_range, upper_range)
            
                #check
                if ((steps + 1) % args.plot_interval == 0) or ((ep == args.num_epoch - 1) and (i == len(train_loader) - 1)):

                    if scheduler is None:
                        lr_now = lr
                    else:
                        lr_now = scheduler.get_last_lr()[0]

                    write_msg = task.eval_model_with_log(model, writer, ep, steps, lr_now, device)

                    stats = ",".join(["{}".format(v) for v in write_msg.values()])
                    f.write(stats + "\n")
                    f.flush()

                    if (args.lr_schedule == "plateau") and (args.task == "cls_CIFAR10"):
                        scheduler.step(write_msg["test_loss"])
                    elif args.lr_schedule == "cosine":
                        scheduler.step()
                    else:
                        pass

                    model.train()

            if args.lr_schedule == "step":
                scheduler.step()
writer.close()
if args.save_model:
    torch.save(model.state_dict(),"code/output_models/{}.pth".format(run_tags))

def local_train(args, task, model, optimizer, device, glb_round):
    pass
    
def update_global_model(args, model, aggregated_local_model_diff):
    pass

def main(args):
    # prepare for logging

    if args.resume is not None:
        run_tags = args.resume.split("/")[-1]
    else:
        ## task tags
        task_tags = f"task:(dt={args.task},md={args.model})"
        
        if args.glb_lr_schedule == "plateau":
            lr_schedule_name = "(plateau,tol:{})".format(args.glb_tol)
        elif args.glb_lr_schedule == "none":
            lr_schedule_name = "none"
        elif args.glb_lr_schedule == "step":
            lr_schedule_name = "(step,ms:{})".format(args.MulStep)

        ## global training tags
        global_tags = f"global_train:(nr={args.num_glb_rounds},nn={args.num_nodes}, \
                                    lr={args.glb_lr},lrs={lr_schedule_name})"



    run_tags = "{}_{}_{}_{}_{}_qi:{}_linf:{}_opt:{}_lrs:{}_bz:{}_wd:{}_lbs:{}_ct:{}".format(args.run_name,
                                                                args.rep,
                                                                args.task,
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

    


    if os.path.exists("code/logs/"+run_tags+".txt") and (not args.rerun) and (args.resume is None):
        print(f"Run {run_tags} already exists. Exit.")
        return
    
    if os.path.exists('code/runs/{}/{}'.format(args.task,run_tags)):
        clean_dir('code/runs/{}/{}'.format(args.task,run_tags))

    if args.tensorboard or True:
        writer = SummaryWriter(logdir='code/runs/{}/{}'.format(args.task,run_tags))
    else:
        writer = None

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        setup_seed(args.seed)

    # define paths
    

    # prepare task and model

    # prepare global optimizer

    # start training
    print("start training:")
    for glb_round in range(args.num_glb_rounds):
        # local training
        aggregated_local_model_diff = local_train(args, task, model, optimizer, device, glb_round)
        global_model = update_global_model(args, model, aggregated_local_model_diff)
        # global evaluation

        # global learning rate scheduling

        # logging and saving


if __name__ == "__main__":
    args = load_args()
    main(args)