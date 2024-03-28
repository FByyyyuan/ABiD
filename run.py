import argparse
from logging import root
import torch
import torch.backends.cudnn as cudnn
import random
from torchvision import models 
from models.backbone import BACKBONE_ABiD
from abid import ABiD
from torch.utils.data import DataLoader
from data_loader.dataloader import read_split_data,SourceDataset,TargetDataset
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ABiD')
parser.add_argument('-root', metavar='DIR', default='/IDRiD/Source',help='path to source dataset')
parser.add_argument('-tarroot', metavar='DIR', default='/IDRiD/Target',help='path to target dataset') 
parser.add_argument('-weightpath', metavar='DIR', default='/params/weightpath.pth', help='path to weight path')
parser.add_argument('-classifier1path', metavar='DIR', default='/params/classifier1.pth', help='path to classifier1 weight path')
parser.add_argument('-classifier2path', metavar='DIR', default='/params/classifier2.pth', help='path to classifier2 weight path')
parser.add_argument('--target', default='T', help='select target domain')
parser.add_argument('--mu', default=0.5, type=float, help='Hyperparameter mu')
parser.add_argument('--txtlog', default='/runs/log.txt',help='recoded by txt') 
parser.add_argument('--val', default=False, type=bool, help='seed for initializing training.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ConvneXt_tiny',
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: ConvneXt_tiny)')
parser.add_argument('-j ', '--workers', default=14, type=int, metavar='N', help='number of data loading workers (default:14)') 
parser.add_argument('--epochs', default=70, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,metavar='N',
                    help='mini-batch size (default: 32), this is the total')
parser.add_argument('--ratio', '--test-set-proportion', default=0, type=float, metavar='RATIO', help='Test set proportion(default = 0)', dest='ratio')
parser.add_argument('--resolution', default=224, type=int, help='resolution for input')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate(default = 0.0001)', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('--logit', default=True, type=bool)
parser.add_argument('--logit_train', default=True, type=bool)
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training.')
parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA') 
parser.add_argument('--fp16-precision', action='store_true',help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=256, type=int, help='representation dimension (default: 256)')
parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.')

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed( seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    set_random_seed(args.seed)
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        args.device = torch.device('cpu')
    
    train_dataset, val = read_split_data(args.root,ratio=args.ratio)
    target_dataset, _ = read_split_data(args.tarroot,ratio=0)

    if args.ratio == 0 and args.val:
        val_dataset, _ =  read_split_data(args.valroot,ratio=args.ratio)
        val = DataLoader(TargetDataset(val_dataset,args=args), batch_size=2*args.batch_size,shuffle=False)
    elif args.ratio != 0:
        val = DataLoader(TargetDataset(val,args=args), batch_size=2*args.batch_size,shuffle=False)
    
    train_loader = DataLoader(SourceDataset(train_dataset,args.root,resolution=args.resolution,tarpath=target_dataset), batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True, drop_last=True)
    target_loader = DataLoader(TargetDataset(target_dataset,args=args,resolution=args.resolution), batch_size=args.batch_size,shuffle=True)
    target_test_loader = DataLoader(TargetDataset(target_dataset,args=args,resolution = args.resolution), batch_size=args.batch_size,shuffle=False)

    model = BACKBONE_ABiD(base_model=args.arch, out_dim=args.out_dim)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    abid = ABiD(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    abid.train(train_loader,target_loader , target_test_loader, test_len = len(target_dataset), val_len=len(val))

if __name__ == "__main__":
    main()