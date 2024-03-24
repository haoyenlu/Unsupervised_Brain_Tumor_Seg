import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import BCELoss, Sigmoid
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime

from tensorboardX import SummaryWriter
import os
import sys

from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg

from dataset import TumorImageDataset
from utils import DiceLoss




vit_name = "ViT-B_16"

config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = 1
config_vit.n_skip = 0


def trainer(model,train_loader,max_epochs,checkpoint_dir,log_dir,checkpoint_path=None):
    logging.basicConfig(filename=log_dir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    writer = SummaryWriter(log_dir + '/log')

    base_lr = 0.1
    num_classes = 1
    #threshold = 0.7

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)

    model.cuda()
    model.train()
    torch.cuda.empty_cache()

    sigmoid = Sigmoid()
    bce_loss = BCELoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0001)

    iter_num = 0
    max_epoch = max_epochs
    max_iterations = max_epoch * len(train_loader)
    iterator = tqdm(range(max_epoch))
    best_performace = 0.0
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))

    for epoch_num in iterator:
        for i_batch, sample_batch in enumerate(train_loader):
            image_batch , label_batch = sample_batch['image'], sample_batch['mask']
            image_batch , label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_bce = bce_loss(sigmoid(outputs),label_batch.unsqueeze(1))
            loss_dice = dice_loss(outputs,label_batch,softmax=True)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            logging.info('iteration %d : loss : %f, loss_bce: %f' % (iter_num, loss.item(), loss_bce.item()))
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_bce', loss_bce, iter_num)
            writer.add_scalar('info/loss_dice',loss_dice,iter_num)


            if iter_num % 20 == 0:
                iterator.set_postfix({"iteration":iter_num,"loss":loss.item(),"loss_bce":loss_bce.item(),"loss_dice":loss_dice.item()})
                sample_image = image_batch[1, 0:1, :, :]
                writer.add_image('train/Image', sample_image, iter_num)
                sample_output = torch.sigmoid(outputs[1,...])
                writer.add_image('train/Prediction', sample_output , iter_num)
                sample_label = label_batch[1, ...].unsqueeze(0)
                writer.add_image('train/GroundTruth', sample_label, iter_num)

            torch.cuda.empty_cache()

        save_mode_path = os.path.join(checkpoint_dir, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = "" #pass in the path of the image folder
    train_dataset = TumorImageDataset(image_dir,is_train=True)
    test_dataset = TumorImageDataset(image_dir,is_train=False)

 
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=True)

    model = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).to(device)
    
    ckpt_path = "" # model pretrained checkpoint
    checkpoint = torch.load(ckpt_path,map_location=device)
    model.load_state_dict(checkpoint)

    save_ckpt_path="" # save model checkpoint
    log_path="" # logging path


    trainer(model,train_dataloader,max_epochs=100,checkpoint_dir=save_ckpt_path,log_dir=log_path)



