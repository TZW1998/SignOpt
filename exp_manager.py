import os
import argparse
import subprocess
import time
from datetime import datetime
from termcolor import colored
import logging

parser = argparse.ArgumentParser(description='Process the exp config.',
                                 conflict_handler='resolve')
parser.add_argument("-f","--file")
parser.add_argument("--file")
parser.add_argument("--gpu_list", nargs='+')
parser.add_argument("--max_jobs_per_gpu", type = int, default = 1)
parser.add_argument("--script")
parser.add_argument("--exp_name")
args = parser.parse_args()




exp_name = args.exp_name + ":" + datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
gpu_list = args.gpu_list
max_jobs_per_gpu = args.max_jobs_per_gpu
script = args.script
file  = args.file


if not os.path.exists("logs"):
    os.mkdir("logs")
    
os.mkdir(os.path.join("logs",exp_name))

logging.basicConfig(filename=os.path.join("./logs",exp_name,"log.log"),
                    filemode='w',
                    datefmt='%m-%d-%Y_%H:%M:%S',
                   level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logger = logging.getLogger()
logger.addHandler(console)


logger.info("Running " + exp_name)




gpu_tasks = {gpu:[] for gpu in gpu_list}
with open(file,"r") as f:
    config = [hp.strip("\n") for hp in f.readlines()]
total_exp_num = len(config)
    
failed_exp = []
finised_exp = 0
while finised_exp < total_exp_num:
    # find empty gpu
    use_gpu = None
    for gpu in gpu_list:
        if len(gpu_tasks[gpu]) < max_jobs_per_gpu:
            use_gpu = gpu
            break
            
    if use_gpu and len(config) > 0:
        # run with gpu
        run_command = "python {} --gpu {} {}".format(script,use_gpu,config.pop())
        logger.info(" ({}/{}) ".format(finised_exp,total_exp_num) +
              " start to execute: " + run_command)
        proc = subprocess.Popen(run_command.split())
        gpu_tasks[use_gpu].append((proc,run_command))
        
    else:
        for _, tasks in gpu_tasks.items():
            finised_jobs = []
            for job_id, (proc,run_command) in enumerate(tasks):
                
                try:
                    proc.communicate(timeout=1)
                except:
                    continue
                    
                if proc.returncode is not None:
                    if proc.returncode == 0:
                        logger.info(" ({}/{}) ".format(finised_exp,total_exp_num) +
                              " The exp {} is finished".format(run_command))
                    else:
                        logger.warning(" ({}/{}) ".format(finised_exp,total_exp_num) +
                              " The exp {} is ".format(run_command) + "failed")
                        failed_exp.append(run_command)
                        
                finised_jobs.append(job_id)
                finised_exp += 1
            
            for job_id in sorted(finised_jobs, reverse=True):
                del tasks[job_id]
        
        time.sleep(1)
        
logger.info(exp_name+" is finished!")
        
failed_exp = [" ".join(exp.split()[4:]) for exp in failed_exp]
with open(os.path.join("logs",exp_name,"failed_exp.txt"),"w") as f:
    for exp in failed_exp:
        f.write(exp)
        f.write("\n")





