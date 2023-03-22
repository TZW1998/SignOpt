import os
from functools import reduce
import argparse

parser = argparse.ArgumentParser(description='Process the hyperparameter file.',
                                 conflict_handler='resolve')
parser.add_argument("-f","--file")
parser.add_argument("--file")
args = parser.parse_args()

file  = args.file
with open(file,"r") as f:
    config = [hp.split(":") for hp in  f.readlines()]
config = {hp[0]:hp[1].strip("\n").split(",") for hp in config}

output_file = file.split(".")[0] + "_output.txt"

total_hp = list(config.keys())
total_hp_len = [len(item) for _,item in config.items()]
total_hp_comb = reduce(lambda a,b: a*b, total_hp_len)

template = " ".join(["--" + hp + " {}" for hp in total_hp])
    
with open(output_file, "w") as f:
    for num in range(total_hp_comb):
        now_index = []
        div = total_hp_comb
        remain_num = num
        for cur_len in total_hp_len:
            div = div// cur_len
            now_index.append(remain_num // div)
            remain_num = remain_num % div
        now_comb = template.format(*[config[total_hp[hp_index]][value_index]
                                     for hp_index,value_index in enumerate(now_index)])
        f.write(now_comb + "\n")