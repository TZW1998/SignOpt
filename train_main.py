from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import numpy as np
import random
import copy
import argparse
import os, shutil
import time
from optimizers import *
from tasks import *
import json
from tqdm import tqdm

TASK_DICT = {"cls_CIFAR10": ClsCIFAR10,
             "diff_MNIST": DiffMNIST,
            }

def load_args():
    parser = argparse.ArgumentParser(description='Process some args.')

    # resume args
    parser.add_argument("--resume", type = str, default = None, help = "to resume from a previous unfinised run")

    # load args from config file
    parser.add_argument("--config", type = str, default = None, help = "to load args from a config file")

    # main experiment args
    parser.add_argument("--gpu", default = "0", help = "gpu id")
    parser.add_argument("--run_name", default = "test", help = "name for this run")
    parser.add_argument("--rerun", type = bool, default = False, help = "whether to rerun this run even though it has been run before")
    parser.add_argument('--seed', type=int, help = "random seed")

    # task args
    parser.add_argument('--task', default = "cls_CIFAR10", choices=["cls_CIFAR10","diff_MNIST"], type=str, help = "task to run")
    parser.add_argument('--model', default = "fastnet", type=str, choices=["resnet18", "resnet18_pretrained", "wide_resnet50_2", "resnet20","fastnet","fastnet2","resnet56P", "unet"], help = "model architecture to use, be careful that some models are only for specific tasks")

    # args for global averaging
    parser.add_argument('--num_glb_rounds', default = 100, type=int, help = "number of global averaging rounds")
    parser.add_argument('--num_nodes', default = 1, type=int, help = "number of parelle nodes")

    # args for global optimizer
    parser.add_argument('--glb_lr', default = 0.1, type=float, help = "learning rate for global averaging")
    parser.add_argument('--glb_optimizer', default = "SGD", type=str, choices=["SGD", "Adam", "AdamW", "SignSGD", "SignFedAvg"])
    parser.add_argument('--glb_beta1', default = 0.9, type=float, help = "For all optimizers, this is related to momentum")
    parser.add_argument('--glb_noise_scale', type=float, default = 0, help = "noise scale for the sign-based compression, only used when optimizer is SignFedAvg")
    parser.add_argument('--glb_sign_allreduce', type=int, default = 0, help = "the way to do all-reduce for signed gradient, 0 means no further compression, 1 means majority vote, n means aggregate the sign by n groups (n should be divided by num_nodes with no remain), only used when optimizer is SignFedAvg")
    parser.add_argument('--glb_beta2', default = 0.99, type=float, help = "For Adam, AdamW, this is about the adaptive step, for SignSGD and SignFedAvg this is related to updated momentum")
    parser.add_argument('--glb_weight_decay', default = 0, type=float, help = "weight decay for global averaging")

    # args for local training
    parser.add_argument('--num_local_steps', default = 1000, type=int, help = "number of local training steps")
    parser.add_argument('--batchsize', default = 32, type=int, help = "batch size for local training")
    parser.add_argument('--lr', default = 0.1, type=float, help = "learning rate for local training, for local training, we use a fixed learning rate")
    parser.add_argument('--optimizer', default = "SGD", type=str, choices=["SGD", "Adam", "AdamW", "SignSGD"])
    parser.add_argument('--beta1', default = 0.0, type=float, help = "For all optimizers, this is related to momentum")
    parser.add_argument('--beta2', default = 0.99, type=float, help = "For Adam, AdamW, and SignSGD, this is about the adaptive step, for SignSGD this is related to updated momentum")
    parser.add_argument('--weight_decay', default = 0, type=float, help = "weight decay for local training")

    # args for logging and saving
    parser.add_argument('--log_tool', default = "tensorboard", type=str, choices=["tensorboard", "wandb", "None"], help = "logging tool")
    parser.add_argument('--plot_interval', default = 1, type=int, help = "interval for logging")
    parser.add_argument('--save_interval', default = 5, type=int, help = "interval for saving")

    # args for torch training/testing
    parser.add_argument('--precison', type=str, default="amp_half", choices=["single", "amp_half"])
    parser.add_argument('--test_batchsize', type=int, default = 1024, help = "batch size for testing")
    parser.add_argument('--physical_batchsize', type=int, default = 1024, help = "physical batch size for training, when physical_batchsize > batchsize, we use gradient accumulation")

    args = parser.parse_args()
    # whether to load other args from a config file
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print("load args from config file: " + args.config)
        for key in config:
            setattr(args, key, config[key])
        print(args)

    # resume from a previous run
    elif args.resume is not None: 
        config_path = os.path.join(args.resume, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("resume from a previous run: " + args.resume)
        for key in config:
                setattr(args, key, config[key])
    else:
        config = { }
        # save the args to a config file
        for key in args.__dict__: 
            if key not in ["gpu", "resume"]:# do not save gpu and resume
                config[key] = args.__dict__[key]

    return args, config


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


def process_tags(args):
    if args.resume is not None:
        return args.resume.split("/")[-1]
    
    ## task tags
    task_tags = f"task:(dt={args.task},md={args.model})"
    
    ## global training tags
    global_tags = f"global_train:(nr={args.num_glb_rounds},nn={args.num_nodes})"
    
    ## global optimizer tags
    global_optimizer_tags = f"global_opt:(opt={args.glb_optimizer}," + \
                                        f"lr={args.glb_lr}," + \
                                        f"wd={args.glb_weight_decay}," + \
                                        f"b1={args.glb_beta1}," + \
                                        f"b2={args.glb_beta2})"
    if args.glb_optimizer == "SignFedAvg":
        global_optimizer_tags += f",ns={args.glb_noise_scale}"
        global_optimizer_tags += f",ar={args.glb_sign_allreduce}"

    ## local optimizer tags
    local_train_tags = f"local_train:(ns={args.num_local_steps}," + \
                                    f"bs={args.batchsize}," + \
                                    f"lr={args.lr}," + \
                                    f"wd={args.weight_decay}," + \
                                    f"b1={args.beta1}," + \
                                    f"b2={args.beta2}," + \
                                    f"opt={args.optimizer})"
    
    # all tags
    run_tags = f"{args.run_name}_seed:{args.seed}_{task_tags}_{global_tags}_{global_optimizer_tags}_{local_train_tags}"
    
    return run_tags

def prepare_logger(args, run_tags):
    if args.log_tool == "tensorboard":
        log_dir = os.path.join("tb_logs", args.task, run_tags)
        if os.path.exists(log_dir) and (args.resume is None):
            clean_dir('tb_logs/{}/{}'.format(args.task,run_tags))
        logger = SummaryWriter(logdir= log_dir)
    elif args.log_tool == "wandb":
        raise Exception(NotImplementedError) # ToDo
    else:
        logger = None

    return logger

def prepare_global_optimizer(args, global_model):
    # global optimizer
    if args.glb_optimizer == "SGD":
        global_optimizer = torch.optim.SGD(global_model.parameters(), lr=args.glb_lr, momentum=args.glb_beta1, weight_decay=args.glb_weight_decay)
    elif args.glb_optimizer == "Adam":
        global_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.glb_lr, betas=(args.glb_beta1, args.glb_beta2), weight_decay=args.glb_weight_decay)
    elif args.glb_optimizer == "AdamW":
        global_optimizer = torch.optim.AdamW(global_model.parameters(), lr=args.glb_lr, betas=(args.glb_beta1, args.glb_beta2), weight_decay=args.glb_weight_decay)
    elif args.glb_optimizer == "SignFedAvg":
        raise Exception(NotImplementedError)
        # ToDo: add SignFedAvg optimizer
        global_optimizer = SignFedAvg(global_model.parameters(), lr=args.glb_lr, betas=(args.glb_beta1, args.glb_beta2), weight_decay=args.glb_weight_decay, noise_scale=args.glb_noise_scale, sign_allreduce=args.glb_sign_allreduce)
    elif args.optimizer == "SignSGD":
        global_optimizer = SignSGD(global_model.parameters(), lr=args.glb_lr, betas=(args.glb_beta1, args.glb_beta2), weight_decay=args.glb_weight_decay)

    return global_optimizer

def prepare_local_optimizer(args, model):
    # global optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.eight_decay)
    elif args.optimizer == "SignSGD":
        optimizer = SignSGD(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    return optimizer

def local_train(args, task, global_model, global_optimizer, device):
    local_models_diff = []
    train_loader = task.get_train_loader()
    if args.num_nodes > 1:
        for _ in range(args.num_nodes):
            model = copy.deepcopy(global_model)
            model.train()
            optimizer = prepare_local_optimizer(args, model)
            train_loader = task.get_train_loader()
            local_steps = 0

            while local_steps < args.num_local_steps:
                for batch_data in train_loader:
                    local_steps += 1
                    optimizer.zero_grad()
                    _ = task.loss_and_step(model, optimizer, batch_data, device)
                    if local_steps == args.num_local_steps:
                        break

            local_model_dict = model.state_dict()
            local_diff = {n : p - local_model_dict[n] for n,p in global_model.named_parameters()}
            local_models_diff.append(local_diff)
    else:
        model = global_model
        optimizer = global_optimizer
        
        model.train()
        local_steps = 0
        while local_steps < args.num_local_steps:
            for batch_data in train_loader:
                local_steps += 1
                optimizer.zero_grad()
                _ = task.loss_and_step(model, optimizer, batch_data, device)
                if local_steps == args.num_local_steps:
                    break

                # if local_steps % 500 == 0:
                #     print(local_steps, task.eval_model_with_log(global_model, device))
                #     model.train()

    return local_models_diff
    
def update_global_model(args, global_model, global_optimizer, local_models_diff):
    global_optimizer.zero_grad()
    if args.glb_optimizer != "SignFedAvg":
        allreduced_gradient = {}
        for n, p in global_model.named_parameters():
            allreduced_gradient[n] = torch.zeros_like(p)
            for local_diff in local_models_diff:
                allreduced_gradient[n] += local_diff[n]
            allreduced_gradient[n] /= len(local_models_diff)
            allreduced_gradient[n] /= args.lr
    else:
        raise Exception(NotImplementedError) # ToDo: add SignFedAvg optimizer
    
    for n, p in global_model.named_parameters():
        p.grad = allreduced_gradient[n]

    global_optimizer.step()

def main(args, config):
    # load tags
    run_tags = process_tags(args)

    # prepare for logging
    log_path = os.path.join("logs", run_tags)
    if os.path.exists(log_path) and (not args.rerun) and (args.resume is None):
        print(f"Run {run_tags} already exists. Exit.")
        return
    else:
        if args.resume is None:
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
            os.makedirs(log_path)
    
        logfile = open(os.path.join(log_path, "log.txt"), "a")
        with open(os.path.join(log_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    # prepare external logger
    logger = prepare_logger(args, run_tags)
    
    # set gpu and seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        setup_seed(args.seed)

    # prepare task and model
    task = TASK_DICT[args.task](batch_size_train = args.batchsize,
                                num_workers = 12,
                                physical_batchsize = args.physical_batchsize,
                                precison = args.precison)
    global_model = task.get_model(args.model).to(device)

    # prepare global optimizer
    global_optimizer = prepare_global_optimizer(args, global_model)

    # path to save model
    model_path = os.path.join(log_path, "model.pth")

    # resume
    if args.resume is not None:
        state_dict = torch.load(model_path)
        start_round = state_dict["round"]
        steps = state_dict["steps"]
        global_model.load_state_dict(state_dict["model"])
        global_optimizer.load_state_dict(state_dict["optimizer"])
        print("resume from {}".format(args.resume))
    else:
        start_round = 0
        steps = 0

    # start training
    print(f"start training for : {run_tags}")
    with tqdm(range(start_round, args.num_glb_rounds), unit="round") as tround:
        for glb_round in tround:
            tround.set_description("round {}".format(glb_round))
            # local training
            steps += args.num_local_steps
            local_models_diff = local_train(args, task, global_model, global_optimizer, device)
            if args.num_nodes > 1: # when args.num_nodes == 1, local training is equivalent to global training
                update_global_model(args, global_model, global_optimizer, local_models_diff)
        
            if glb_round % args.plot_interval == 0:
                # global evaluation
                log_msg = task.eval_model_with_log(global_model, device)
                # get current info
                log_msg["glb_round"] = glb_round
                log_msg["glb_steps"] = steps

                # logging 
                stats = ",".join(["{}".format(v) for v in log_msg.values()])
                logfile.write(stats + "\n")
                logfile.flush()
                tround.set_postfix(log_msg)

                # log to external logger
                if args.log_tool == "tensorboard":
                    for k, v in log_msg.items():
                        logger.add_scalar("stats/" + k, v, steps)
                elif args.log_tool == "wandb":
                    raise NotImplementedError # ToDo: add wandb logger

            # saving
            if glb_round % args.save_interval == 0:
                print("saving model at round {}".format(glb_round))
                torch.save({"glb_round":glb_round,
                            "gradient_steps": steps,
                            "model": global_model.state_dict(),
                            "optimizer": global_optimizer.state_dict()},
                            model_path)
                
    # end logging
    logfile.close()
    if logger is not None:
        logger.close()

    # save final model
    final_model_path =  os.path.join(log_path, "final_model.pth")
    torch.save(global_model.state_dict(), final_model_path)

if __name__ == "__main__":
    args, config = load_args()
    main(args, config)