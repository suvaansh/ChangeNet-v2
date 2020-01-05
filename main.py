import argparse
import torch 
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import csv
import os
import math
from PIL import Image
import numpy as np
from engine import Engine
from dataloader import *
from model import Model
from utils import *

parser = argparse.ArgumentParser(description='VL_CMU_CD')

parser.add_argument('--data', metavar='DIR',default='../../../VL_CMU_CD', help='path to dataset (e.g. ../data/')
parser.add_argument('--image_size', '-i', default=(192,256), help='image size (default: (192,256))')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=0, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pos_weights', default=(0.2, 0.8), help='class weights due to dataset imbalance')
parser.add_argument('-v', '--efile', default='val', type=str,  help='evaluation csv file')


# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TORCH_HOME"] = "../MODEL"

def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    args.resume = os.path.join('./models', args.resume)  ####################################################################################################################

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = GSV(args.data, 'train')
    val_dataset = GSV(args.data, args.efile)

    # load model
    backbone = torchvision.models.vgg16(pretrained=True)
    model = Model(args.image_size[0],args.image_size[1], backbone)
    print("Model Created")
    
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr)

    state = {'batch_size': args.batch_size, 'train_image_size': args.image_size, 'test_image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume}
    state['save_model_path'] = './models'
    state['epoch_step']={0}
    state['print_freq'] = args.print_freq
    state['pos_weight_neg'] = args.pos_weights[0]
    state['pos_weight_pos'] = args.pos_weights[1]
    state['threshold'] = 0.5
    state['multi_gpu'] = True

    state['CATEGORY_TO_LABEL_DICT'] = {'background': 0, 'barrier': 1, 'bin': 2, 'construction': 3, 'person/bicycle': 4, 'rubbish_bin': 5, 'sign_board': 6, 'traffic_cone': 7, 'vehicles': 8, 'other_objects': 9,}
    state['LABEL_TO_CATEGORY_DICT'] = {v: k for k, v in state['CATEGORY_TO_LABEL_DICT'].items()}

    engine = Engine(state)
    engine.learning(model, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main()
