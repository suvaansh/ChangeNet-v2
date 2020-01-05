import os
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from utils import AveragePrecisionMeter, Warp
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn.functional as F

class Engine(object):
    def __init__(self, state={}):

        self.state = state

        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('train_image_size') is None:
            self.state['train_image_size'] = (192, 256)
        if self._state('val_image_size') is None:
            self.state['val_image_size'] = (192, 256)

        if self._state('batch_size') is None:
            self.state['batch_size'] = 1

        if self._state('train_workers') is None:
            self.state['train_workers'] = 16
        if self._state('val_workers') is None:
            self.state['val_workers'] = 4

        if self._state('multi_gpu') is None:
            self.state['multi_gpu'] = True

        if self._state('device_ids') is None:
            if self.state['evaluate']:
                print("Evaluating on single GPU")
                self.state['device_ids'] = [0]  
            else:
                self.state['device_ids'] = [0]

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 100

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        if self.state['pos_weight_neg'] is None:
        	self.state['pos_weight_neg'] = 0.2
        if self.state['pos_weight_pos'] is None:
        	self.state['pos_weight_pos'] = 0.8

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        self.state['ap_meter'] = AveragePrecisionMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

        # self.writer = SummaryWriter()

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def learning(self, model, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['train_workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['val_workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                self.state['n_iter'] = checkpoint['n_iter']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        # print("############################################################################################################1")
        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            if self.state['multi_gpu']:
                model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
                # print("5and7")
            else:
                model = torch.nn.DataParallel(model).cuda()
                print("normal")
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        if self.state['evaluate']:
            with torch.no_grad():
                self.validate(val_loader, model)
            return


        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, optimizer, epoch)

            # evaluate on validation set
            with torch.no_grad():
                prec1 = self.validate(val_loader, model)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
                'n_iter' : self.state['n_iter']
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']), '\n')


    def init_learning(self, model):

        normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                 std=model.image_normalization_std)
        self.state['train_transform'] = transforms.Compose([
            transforms.ToTensor()])
        self.state['val_transform'] = transforms.Compose([
            transforms.ToTensor()])
        self.state['train_target_transform'] = transforms.Compose([
            transforms.ToTensor()])
        self.state['val_target_transform'] = transforms.Compose([
            transforms.ToTensor()])
        self.state['best_score'] = 0

        self.state['n_iter'] = 0
        
        # self.state['pos_weight'] = (self.state['pos_weight_neg']*(self.state['target'] ==0).float() + self.state['pos_weight_pos']*(self.state['target'] ==1).float())

        self.state['pos_weight'] = torch.ones([10])*self.state['pos_weight_pos']
        self.state['pos_weight'][0] = self.state['pos_weight_neg']

        # (self.state['pos_weight_neg']*(self.state['target'] ==0).float() + self.state['pos_weight_pos']*(self.state['target'] ==1).float())

        self.state['criterion'] = nn.CrossEntropyLoss(self.state['pos_weight'])


    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        if self.state['epoch'] is not 0 and self.state['epoch'] in self.state['epoch_step']:
            print('update learning rate')
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = param_group['lr'] * 0.1
                print(param_group['lr'])

            self.state['resume'] = 'model_best.pth.tar'
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                # self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

    def train(self, data_loader, model, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (img1, img2, target, mask) in enumerate(data_loader):

            # print(target[0].shape, mask[0].shape, img1[0].shape, img2[0].shape)
            # print(np.unique(target[0].cpu().numpy()))
            # exit()

            # if i>1:
            #     continue

            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['img1'] = img1
            self.state['img2'] = img2
            self.state['target'] = target
            self.state['mask'] = mask

            self.on_start_batch(True, model, data_loader, optimizer)  

            
            # self.state['pos_weight'] = (self.state['pos_weight_neg']*(self.state['target'] ==0).float() + self.state['pos_weight_pos']*(self.state['target'] ==1).float())

            # print(self.state['target'] , '\n', self.state['target'] ==0, '\n', self.state['pos_weight'])

            if self.state['use_gpu']:
                self.state['img1'] = self.state['img1'].cuda()
                self.state['img2'] = self.state['img2'].cuda()
                self.state['target'] = self.state['target'].cuda()
                self.state['mask'] = self.state['mask'].cuda()
                # self.state['pos_weight'] = self.state['pos_weight'].cuda()
                self.state['criterion'] = self.state['criterion'].cuda()

            self.on_forward(True, model, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, data_loader, optimizer)

        self.on_end_epoch(True, model, data_loader, optimizer)

    def validate(self, data_loader, model):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (img1, img2, target, mask) in enumerate(data_loader):
            
            # if i>1:
            #     continue

            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            if self.state['iteration'] >0:
                self.state['data_time'].add(self.state['data_time_batch'])

            self.state['img1'] = img1
            self.state['img2'] = img2
            self.state['target'] = target
            self.state['mask'] = mask

            self.on_start_batch(False, model, data_loader)

            # self.state['pos_weight'] = (self.state['pos_weight_neg']*(self.state['target'] ==0).float() + self.state['pos_weight_pos']*(self.state['target'] ==1).float())

            
            if self.state['use_gpu']:
                self.state['img1'] = self.state['img1'].cuda()
                self.state['img2'] = self.state['img2'].cuda()
                self.state['target'] = self.state['target'].cuda()
                self.state['mask'] = self.state['mask'].cuda()
                # self.state['pos_weight'] = self.state['pos_weight'].cuda()
                self.state['criterion'] = self.state['criterion'].cuda()

            self.on_forward(False, model, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            if self.state['iteration'] >0:
                self.state['batch_time'].add(self.state['batch_time_current'])

            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, data_loader)

        score = self.on_end_epoch(False, model, data_loader)

        return score

    def on_start_epoch(self, training, model, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()
        self.state['ap_meter'].reset()


    def on_start_batch(self, training, model, data_loader, optimizer=None, display=True):

        img1 = self.state['img1']
        self.state['img1'] = img1[0].float()
        # print(self.state['img1'].shape, self.state['img1'])
        self.state['img1_path'] = img1[1]

        img2 = self.state['img2']
        self.state['img2'] = img2[0].float()
        # print(self.state['img2'].shape, self.state['img2'])
        self.state['img2_path'] = img2[1]
        
        target = self.state['target']
        self.state['target'] = target[0].float()
        # print(self.state['target'].shape, self.state['target'])
        self.state['target_path'] = target[1]

        mask = self.state['mask']
        self.state['mask'] = mask[0].float()
        # print(self.state['target'].shape, self.state['target'])
        self.state['mask_path'] = mask[1]


    def on_forward(self, training, model, data_loader, optimizer=None, display=True):

        # compute output
        img1 = (self.state['img1'])
        img2 = (self.state['img2'])
        target = (self.state['target'])
        mask = (self.state['mask'])

        self.state['output'] = model(img1, img2)
        self.state['pred'] = F.softmax(self.state['output'], dim=1)
        
        self.state['thresh_pred'] =  torch.argmax(self.state['pred'], dim=1)
        
        # print(np.unique(self.state['pred'].data.cpu().numpy()))
        # print('thresh', np.unique(self.state['thresh_pred'].data.cpu().numpy()), torch.sum(self.state['thresh_pred']))
        # print('target', np.unique(self.state['target'].data.cpu().numpy()), torch.sum(self.state['target']))
       	# print(self.state['output'].shape, mask.shape, target.shape)
        # print("mask.shape, target.shape", mask.shape, target.shape, (mask*target).squeeze(1).shape)
        self.state['loss'] = self.state['criterion']((mask*self.state['output']), (mask*target).squeeze(1).long())

        # if training:
        #     self.writer.add_image('train/Image1', vutils.make_grid(img1, normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('train/Image2', vutils.make_grid(img2, normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('train/target', vutils.make_grid(target, normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('train/thresh_prediction', vutils.make_grid(self.state['thresh_pred'].float(), normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('train/prediction', vutils.make_grid(self.state['pred'], normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('train/output', vutils.make_grid(self.state['output'], normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_scalar('train/loss', self.state['loss'], self.state['n_iter'])
        # else:
        #     self.writer.add_image('val/Image1', vutils.make_grid(img1, normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('val/Image2', vutils.make_grid(img2, normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('val/target', vutils.make_grid(target, normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('val/thresh_prediction', vutils.make_grid(self.state['thresh_pred'].float(), normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('val/prediction', vutils.make_grid(self.state['pred'], normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_image('val/output', vutils.make_grid(self.state['output'], normalize=True, scale_each=True), self.state['n_iter'])
        #     self.writer.add_scalar('val/loss', self.state['loss'], self.state['n_iter'])

        self.state['n_iter'] = self.state['n_iter']+1

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()


    def on_end_batch(self, training, model, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].data
        self.state['meter_loss'].add(self.state['loss_batch'].cpu())
        self.state['ap_meter'].add(self.state['thresh_pred'], self.state['output'].data, self.state['target'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
                
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time),'Loss loss_current:' + str(self.state['loss_batch'].cpu().numpy()) + ' loss:' + str(loss.cpu().numpy()))
                f = open("eval_logs.csv", 'a')
                
                write_string = 'Test: [{0}/{1}]\t Time {batch_time_current:.3f} ({batch_time:.3f})\t Data {data_time_current:.3f} ({data_time:.3f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time) + (str(self.state['loss_batch'].cpu().numpy()) + ' loss:' + str(loss.cpu().numpy()) + '\n')

                f.write(write_string)
                f.close()

    def on_end_epoch(self, training, model, data_loader, optimizer=None, display=True):

        TPs, FPs, TNs, FNs = self.state['ap_meter'].value_metrics()

        TPs = TPs[1:]
        FPs = FPs[1:]
        TNs = TNs[1:]
        FNs = FNs[1:]

        TP = TPs.sum()
        FP = FPs.sum()
        TN = TNs.sum()
        FN = FNs.sum()
        precision = TP/max((TP+FP),1)
        recall = TP/max((TP+FN),1)
        f1_score = 2*precision*recall/max((precision + recall),1)
        
        APs = 100 * self.state['ap_meter'].value()
        APs = APs[1:]
        
        map = APs.mean()

        loss = self.state['meter_loss'].value()[0]


        # Unions = TPs + FPs + FNs
        # Intersections = TPs
        # IOUs = Intersections / Unions

        CATEGORY_TO_LABEL_DICT = self.state['CATEGORY_TO_LABEL_DICT']
        LABEL_TO_CATEGORY_DICT = self.state['LABEL_TO_CATEGORY_DICT']

        if display:
            if training:
                
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}\t'
                      'TP {TP:.0f}\t'
                      'FP {FP:.0f}\t'
                      'TN {TN:.0f}\t'
                      'FN {FN:.0f}\t'
                      'prec {prec:.3f}\t'
                      'rec {rec:.3f}\t'
                      'f1 {f1:.3f}'.format(self.state['epoch'], loss=loss.cpu().numpy(), map=map, TP=TP, FP=FP, TN=TN, FN=FN, prec=precision, rec=recall, f1=f1_score))

                if not self.state['evaluate']:
                    f = open("train_logs.csv", 'a')
                # else:
                #    f = open("eval_logs.csv", 'a')

                write_string = '\n{epoch:.0f}\t Train:\t {loss:.4f}\t {mAP:.3f} \t\t'\
                  .format(epoch = self.state['epoch'], loss=loss.cpu().numpy(), mAP=map )
                f.write(write_string)
                f.close()

            else:
                if not self.state['evaluate']:
                    print('\t\t'
                          'Loss {loss:.4f}\t'
                          'mAP {map:.3f}\t'
                          'TP {TP:.0f}\t'
                          'FP {FP:.0f}\t'
                          'TN {TN:.0f}\t'
                          'FN {FN:.0f}\t'
                          'prec {prec:.3f}\t'
                          'rec {rec:.3f}\t'
                          'f1 {f1:.3f}'.format(loss=loss.cpu().numpy(), map=map, TP=TP, FP=FP, TN=TN, FN=FN, prec=precision, rec=recall, f1=f1_score))
                    
                    
                    f = open("train_logs.csv", 'a')
                    write_string = 'Test:\t {loss:.4f}\t {mAP:.3f}\t {TP:.0f}\t {FP:.0f}\t {TN:.0f}\t {FN:.0f}\t {prec:.3f}\t {rec:.3f}\t {f1:.3f}'\
                        .format(loss=loss.cpu().numpy(), mAP=map, TP=TP, FP=FP, TN=TN, FN=FN, prec=precision, rec=recall, f1=f1_score)
                    f.write(write_string)
                    f.close()

                else:
                    f = open("eval_logs.csv", 'a')
                    model_name = self._state('resume')
                    write_string_title = f'\n\n{model_name}\t'
                    write_string_prec = f'\nPrecision:\t'
                    write_string_rec = f'\nRecall:\t'
                    write_string_f1 = f'\nF1 Score:\t'

                    for ind in range(len(TPs)):
                        tp = TPs[ind]
                        fp = FPs[ind]
                        tn = TNs[ind]
                        fn = FNs[ind]
                        prec = tp/max((tp+fp),1)
                        rec = tp/max((tp+fn),1)
                        f1 = 2*prec*rec/max((prec + rec),1)
                        write_string_title += f'{LABEL_TO_CATEGORY_DICT[ind+1]}\t'
                        write_string_prec += f'{prec:.3}\t'
                        write_string_rec += f'{rec:.3}\t'
                        write_string_f1 += f'{f1:.3}\t'

                    write_string_title += f'Overall\t'
                    write_string_prec += f'{precision:.3}\t'
                    write_string_rec += f'{recall:.3}\t'
                    write_string_f1 += f'{f1_score:.3}\t'


                    write_string = write_string_title + write_string_prec + write_string_rec + write_string_f1
                    # print(write_string)
                    f.write(write_string)
                    f.close()
        # self.writer.close()
        return f1_score


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
               # if self._state('filename_previous_best') is not None:
                   # os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    
    # def write_summary_img(self, name, tensor, n_iter):
    #     self.writer.add_image(name, vutils.make_grid(tensor), n_iter)
