import sys
import itertools
import subprocess
import torch

lr_multipliers = [0.1, 0.5, 1, 3, 5]
base_lrs = [0.0001, 0.00001]

cmd_ = 'python main.py'
cmd_ += ' --max_iter 10000'
cmd_ += ' --glr %s --dlr %s'
cmd_ += ' --name %s'

cmds = []
prod = list(itertools.product(base_lrs, lr_multipliers))
for params in prod:
    base_lr, lr_multiplier = params

    glr = base_lr
    dlr = glr*lr_multiplier
    #name = 'run_glr:%.6f_dlr:%.6f' % (glr, dlr) 
    name = 'run_glr:%.6f_dlr:%.6f_notanh' % (glr, dlr) 
    cmd = cmd_ % (str(glr), str(dlr), name)
    cmds.append(cmd)

#splits = torch.arange(len(cmds)).chunk(8)
splits = torch.arange(len(cmds)).split(1)
splits = splits[int(sys.argv[1])]
print(len(splits))
for split_idx, split in enumerate(splits):
    print(cmds[split])
    # print(log_dirs[split])
    try:
        subprocess.call(cmds[split].split())
    except KeyboardInterrupt as e:
       # sys.exit(-1)
       pass

