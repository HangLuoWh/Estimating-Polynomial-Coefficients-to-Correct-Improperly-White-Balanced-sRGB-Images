import numpy as np
import torch
import torch.nn as nn
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils
from Dataset import TrainSet, ValiSet
from Hyper import hyper
from Polynomial_Net import PolyNet
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

def train_net(net, train_part, vali_part, writer, check_point = None):
    """
    model training
    parameters:
        net: model
        train_part traning dataset
        vali_part: validation dataset
        writer: tensorboard writer
        chech_point: 
    """
    train_loader = DataLoader(train_part, hyper['batch_size'], shuffle=True, \
        num_workers = 4, pin_memory=True)
    
    test_batch = next(iter(train_loader)) # one batch for validation

    if check_point is None: # start training from the scratch
        opti_g = torch.optim.Adam(net.parameters(), lr = hyper['lr'],\
            betas=(hyper['b1'], hyper['b2']))
        scheduler_g = utils.define_scheduler(opti_g, hyper) # 定义调度器
        start_epoch = 0
        global_step = 0
    else: # start training from a check point
        net.load_state_dict(check_point['model'])
        opti_g = torch.optim.Adam(net.parameters(), lr = hyper['lr'],\
            betas=(hyper['b1'], hyper['b2']))
        opti_g.load_state_dict(check_point['optimizer'])
        scheduler_g = utils.define_scheduler(opti_g, hyper) # 定义调度器
        scheduler_g.load_state_dict(check_point['scheduler']) 
        start_epoch = check_point['epoch'] + 1
        global_step = check_point['global_step'] + 1
        hyper['best_vali_loss'] = check_point['best_vali_loss']

    loss_f = nn.L1Loss().to(hyper['dev']) # loss function
    
    # start iteration
    for epoch in range(start_epoch, hyper['epochs']):
        net.train()
        loss_epoch = 0 # the average loss of an epoch
        batch_num = len(train_loader) # the number of batches of an epoch
        with tqdm(total=batch_num, leave=True, \
            desc=f'epoch {epoch}/{hyper["epochs"]-1}', unit='it') as process_bar:
            for _, batch in enumerate(train_loader):
                wrong_img = batch['input'].to(hyper['dev']) # input
                gt_img = batch['gt'].to(hyper['dev']) # ground truth
                batch_loss = 0 # loss of on batch
                
                # forward-backward
                for i in range(hyper['patch_num']):
                    img_p = wrong_img[:, (i*3):3+(i*3), :, :] # 输入图像块
                    gt_p = gt_img[:, (i*3):3+(i*3), :, :] # 真实图像块
                    p_pre, para = net(img_p) # 正向传播
                    p_loss = loss_f(p_pre, gt_p)#计算模型损失
                    batch_loss += p_loss.item()
                    opti_g.zero_grad()
                    p_loss.backward()
                    opti_g.step()
                
                batch_loss /= hyper['patch_num']
                loss_epoch += batch_loss # 将batch的损失计入epoch损失中

                writer.add_scalar('batch', batch_loss, global_step)
                process_bar.set_postfix({'batch':batch_loss})
                process_bar.update(1)
                global_step += 1
        
        scheduler_g.step()
        writer.add_scalar('epoch', loss_epoch/batch_num, epoch)
        print(f'''epoch:{epoch}, current_loss:{loss_epoch/batch_num}, current_best:{hyper['best_vali_loss']}''')
        
        # for the first epoch, show the test batch
        if epoch == 0:
            writer.add_image('test', make_grid(test_batch['input'][:,0:3,:,:], \
                nrow=20, padding=0), epoch)
            writer.add_image('test_gt', make_grid(test_batch['gt'][:,0:3,:,:], \
                nrow=20, padding=0), epoch)

        # save the training state
        if (epoch+1) % hyper['save_fre'] == 0:
            utils.save_model(net, opti_g, scheduler_g, hyper, global_step, epoch, False) # 存储模型
            print(f'epoch: {epoch} model saved')

        # validate the model
        if (epoch+1) % hyper['vali_fre'] == 0:
            # correct images in the test_batch
            net.eval()
            with torch.no_grad():
                trans_img, _ = net(test_batch['input'][:, 0:3, :, :].to(hyper['dev']))
            net.train()

            writer.add_image('gen_img', make_grid(trans_img.detach(), nrow=20, padding=0), epoch) # show generated images
            writer.add_scalar('learning_rate', opti_g.param_groups[0]['lr'], epoch) # show learning rate
            
            vali_loss = vali_net(net, vali_part) # validate on the valdiate set
            writer.add_scalar('loss_vali', vali_loss, epoch) # show validation loss
            # save the model with the lowest validation loss
            if (hyper['best_vali_loss'] is None) or (vali_loss < hyper['best_vali_loss']):
                hyper['best_vali_loss'] = vali_loss
                utils.save_model(net, opti_g, scheduler_g, hyper, global_step, epoch, best = True)

            # show parameter distributions of the model
            for name, para in net.named_parameters():
                writer.add_histogram(name, para, global_step = epoch)

# valdiate the model
def vali_net(net, dataset):
    """
    parameter:
        net: model
        dataset: valdiation dataset
    """
    net.eval() #
    loader = DataLoader(dataset, batch_size = hyper['vali_batch_size'], shuffle=True, \
        num_workers = 4, pin_memory=True) # validation dataloader
    n_val = len(loader)
    mae = 0
    loss_f = nn.L1Loss().to(hyper['dev']) # validation loss
    with tqdm(total=n_val, desc='valdiation', unit='it', leave=False) as pbar:
        for batch in loader:
            wrong_imgs = batch['input'].to(hyper['dev'])
            gt_imgs = batch['gt'].to(hyper['dev'])

            with torch.no_grad():# don't track gradients
                p_pre = net(wrong_imgs)
                loss = loss_f(p_pre, gt_imgs)
                mae += loss.item()
            
            pbar.update(1)
            pbar.set_postfix({'vali_loss':loss.item()})
    net.train()
    return  mae/n_val

if __name__ == "__main__":
    # lock random seeds
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # dataset setting
    ds_dir = './train_set/set1_no_chart' # the path of traing set
    set1_input = os.path.join(ds_dir, 'input') # the path of input images in the traing set
    set1_gt = os.path.join(ds_dir, 'ground_truth') # the path of ground truth images in the traing set
    train_ls, vali_ls = utils.load_train_vali_ls(ds_dir) # load training and validation image lists

    if not os.path.isdir(hyper['check_dir']): os.makedirs(hyper['check_dir'], exist_ok=True) # create a directory to store check points

    # set writer directory to store training histories
    writer_dir = './results/'
    writer_dir += 'p' if not hyper['root'] else 'r'
    writer_dir += str(hyper['term'])
    writer_dir += '_'
    writer_dir += 'no_res_' if not hyper['residual'] else 'res_'
    writer_dir += 'no_na_' if not hyper['neuron_attention'] else 'na_'
    writer_dir += 'no_ca_' if not hyper['channel_attention'] else 'ca_'
    writer_dir += str(hyper['linear_num'])
    hyper['writer'] = SummaryWriter(writer_dir)
    if not os.path.isdir(writer_dir): os.makedirs(writer_dir, exist_ok=True) # 创建保存节点的文件夹

    train_set = TrainSet(train_ls, set1_input, set1_gt, \
        patch_size = hyper['patch_size'], patch_num = hyper['patch_num']) # create the train set
    vali_set = ValiSet(vali_ls, set1_input, set1_gt,\
        patch_size = hyper['patch_size']) # create the validation set

    check = None # check point directore. None represents retraining.

    net = PolyNet(hyper).to(hyper['dev'])
    print(f'''root={hyper['root']}, term={hyper['term']}, residual={hyper['residual']},
        neuron_attention={hyper['neuron_attention']}, channel_attention={hyper['channel_attention']},
        linear_num = {hyper['linear_num']}''')
    train_net(net, train_set, vali_set, hyper['writer'], check) # 训练模型