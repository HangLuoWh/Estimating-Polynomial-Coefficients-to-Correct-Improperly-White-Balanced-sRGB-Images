import numpy as np
import os
import cv2
import os
import torch
from torch.optim import lr_scheduler

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:,0], I[:,1], I[:,2], I[:,0] * I[:,1], I[:,0] * I[:,2],
                          I[:,1] * I[:,2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1,np.shape(I)[0]))))

def get_gt_name(input_name):
    """
    根据输入图片的名称，得到真实图片的名称
    参数：
        input_name:输入图片的名称
    """
    seperator = '_'
    temp = input_name.split(seperator)
    gt_file = seperator.join(temp[:-2])
    gt_file = gt_file + '_G_AS.png'
    return gt_file

def load_train_vali_ls(path):
    """
    load training and validation image lists
    parameter:
        path: the path storing 'train.txt', 'vali.txt'
    """
    train = [line.strip('\n')
             for line in open(os.path.join(path, 'train.txt'))]
    vali = [line.strip('\n')
             for line in open(os.path.join(path, 'vali.txt'))]
    return train, vali

def load_test_ls(path):
    """
    load test image list
    """
    return [line.strip('\n')
            for line in open(os.path.join(path, 'test.txt'))]

def img_trans(img, para):
    """
    在已知转换参数的情况下，对图像的颜色进行转换
    参数： 
        img: 待转换图像；
        para: 转换参数
    """
    img_shape = img.shape
    img_t = np.reshape(img, (img_shape[0]*img_shape[1], img_shape[2]))
    img_kernel = kernelP(img_t)

    para = np.reshape(para, (3, 11))
    # 对图像进行转换
    trans_img = np.matmul(img_kernel, para.T)
    rlt = np.reshape(trans_img, (img_shape[0], img_shape[1], img_shape[2]))
    # 进行数值的裁切
    rlt[rlt > 1] = 1 # any pixel is higher than 1, clip it to 1
    rlt[rlt < 0] = 0 # any pixel is below 0, clip it to 0
    return rlt

def tensor_trans(tensor, paras):
    """
    对一个张量内所有的图像进行批量的颜色转换
    """
    input_array = tensor.numpy() # 将图片张量转为array
    # 改变维度的顺序
    input_array = np.transpose(input_array, (0, 2, 3, 1)) # 张量轴顺序从CHW转为HWC
    paras = paras.cpu().detach().numpy() # 将转换参数转为array
    total = input_array.shape[0] # 总图像数量
    rlt = np.ones_like(input_array) # 创建一个放置运算结果的array
    # 遍历所有图像，并且进行图像转换
    for i in range(total):
        img = input_array[i, :, :, :]
        img_para = paras[i, :]
        rlt[i, :, :, :] = img_trans(img, img_para) # 对图像进行转换
    rlt = np.tensor(np.transpose(rlt, (0, 3, 1, 2))) # HWC to CHW
    return rlt

def im2double(im):
    """ Returns a double image [0,1] of the uint8 im [0,255]. """
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# 对输入张量进行多项式拓展
def extend(img, num, root=False):
    """
    对图像进行多项式拓展
    参数：
        img: 图像
        type: 多项式类型，如果为True，就是根多项式拓展；如果为False，就是普通多项式拓展
        num： 拓展项数。root为false, 可选：3，5，6，8，9，11；root为True， 可选： 6, 13, 22
    """
    if not root:
        if num == 3:
            return img.transpose((2, 0, 1)) # 将通道维度放在第一个维度上
        elif num == 4:
            shift = 0.000001
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_lr = np.log10(img_r + shift)
            img_lg = np.log10(img_g + shift)
            img_lb = np.log10(img_b + shift)

            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_lr = np.expand_dims(img_lr, 0)
            img_lg = np.expand_dims(img_lg, 0)
            img_lb = np.expand_dims(img_lb, 0)

            return np.concatenate([img_r, img_g, img_b, \
                img_lr, img_lg, img_lb])
                
        elif num == 5:
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rgb = img_r*img_g*img_b
            img_1 = np.ones_like(img_r)

            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rgb = np.expand_dims(img_rgb, 0)
            img_1 = np.expand_dims(img_1, 0)

            return np.concatenate([img_r, img_g, img_b, \
                img_rgb, img_1])

        elif num == 6:
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b

            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)

            return np.concatenate([img_r, img_g, img_b, \
                img_rg, img_rb, img_gb])
        elif num == 7:
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_1 = np.ones_like(img_r)

            # 添加新的维度
            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)
            img_1 = np.expand_dims(img_1, 0)

            return np.concatenate([img_r, img_g, img_b, \
                img_rg, img_rb, img_gb, img_1])
        elif num == 9:
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_r2 = img_r*img_r
            img_g2 = img_g*img_g
            img_b2 = img_b*img_b

            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)
            img_r2 = np.expand_dims(img_r2, 0)
            img_g2 = np.expand_dims(img_g2, 0)
            img_b2 = np.expand_dims(img_b2, 0)

            return np.concatenate([img_r, img_g, img_b, \
                img_rg, img_rb, img_gb, img_r2, img_g2, img_b2])
        elif num == 11:
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_r2 = img_r*img_r
            img_g2 = img_g*img_g
            img_b2 = img_b*img_b
            img_rgb = img_r*img_g*img_b
            img_1 = np.ones_like(img_r)

            # 添加新的维度
            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)
            img_r2 = np.expand_dims(img_r2, 0)
            img_g2 = np.expand_dims(img_g2, 0)
            img_b2 = np.expand_dims(img_b2, 0)
            img_rgb = np.expand_dims(img_rgb, 0)
            img_1 = np.expand_dims(img_1, 0)

            return np.concatenate([img_r, img_g, img_b, \
                img_rg, img_rb, img_gb, img_r2, img_g2, img_b2, \
                img_rgb, img_1])

        elif num == 19: # 多项式项数为19项时
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_r2 = np.power(img_r, 2)
            img_g2 = np.power(img_g, 2)
            img_b2 = np.power(img_b, 2)
            img_r3 = np.power(img_r, 3)
            img_g3 = np.power(img_g, 3)
            img_b3 = np.power(img_b, 3)
            img_rg2 = img_r*img_g**2
            img_gb2 = img_g*img_b**2
            img_rb2 = img_r*img_b**2
            img_gr2 = img_g*img_r**2
            img_bg2 = img_b*img_g**2
            img_br2 = img_b*img_r**2
            img_rgb = img_r * img_g * img_b

            # 添加新的维度
            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)
            img_r2 = np.expand_dims(img_r2, 0)
            img_g2 = np.expand_dims(img_g2, 0)
            img_b2 = np.expand_dims(img_b2, 0)
            img_r3 = np.expand_dims(img_r3, 0)
            img_g3 = np.expand_dims(img_g3, 0)
            img_b3 = np.expand_dims(img_b3, 0)
            img_rg2 = np.expand_dims(img_rg2, 0)
            img_gb2 = np.expand_dims(img_gb2, 0)
            img_rb2 = np.expand_dims(img_rb2, 0)
            img_gr2 = np.expand_dims(img_gr2, 0)
            img_bg2 = np.expand_dims(img_bg2, 0)
            img_br2 = np.expand_dims(img_br2, 0)
            img_rgb = np.expand_dims(img_rgb, 0)

            return np.concatenate([img_r, img_g, img_b, img_r2, img_g2, img_b2, img_rg, img_gb, \
                                img_rb, img_r3, img_g3, img_b3, img_rg2, img_gb2, img_rb2, \
                                img_gr2, img_bg2, img_br2, img_rgb])
        elif num == 34: # 多项式项数为34项时
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_r2 = np.power(img_r, 2)
            img_g2 = np.power(img_g, 2)
            img_b2 = np.power(img_b, 2)
            img_r3 = np.power(img_r, 3)
            img_g3 = np.power(img_g, 3)
            img_b3 = np.power(img_b, 3)
            img_rg2 = img_r*img_g**2
            img_gb2 = img_g*img_b**2
            img_rb2 = img_r*img_b**2
            img_gr2 = img_g*img_r**2
            img_bg2 = img_b*img_g**2
            img_br2 = img_b*img_r**2
            img_rgb = img_r * img_g * img_b

            img_r4 = img_r**4
            img_g4 = img_g**4
            img_b4 = img_b**4
            img_r3g = img_g*img_r**3
            img_r3b = img_b*img_r**3
            img_g3r = img_r*img_g**3
            img_g3b = img_b*img_g**3
            img_b3r = img_r*img_b**3
            img_b3g = img_g*img_b**3
            img_r2g2 = img_r**2*img_g**2
            img_g2b2 = img_g**2*img_b**2
            img_r2b2 = img_r**2*img_b**2
            img_r2gb = img_r**2*img_g*img_b
            img_g2rb = img_g**2*img_r*img_b
            img_b2rg = img_b**2*img_r*img_b

            # 添加新的维度
            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)
            img_r2 = np.expand_dims(img_r2, 0)
            img_g2 = np.expand_dims(img_g2, 0)
            img_b2 = np.expand_dims(img_b2, 0)
            img_r3 = np.expand_dims(img_r3, 0)
            img_g3 = np.expand_dims(img_g3, 0)
            img_b3 = np.expand_dims(img_b3, 0)
            img_rg2 = np.expand_dims(img_rg2, 0)
            img_gb2 = np.expand_dims(img_gb2, 0)
            img_rb2 = np.expand_dims(img_rb2, 0)
            img_gr2 = np.expand_dims(img_gr2, 0)
            img_bg2 = np.expand_dims(img_bg2, 0)
            img_br2 = np.expand_dims(img_br2, 0)
            img_rgb = np.expand_dims(img_rgb, 0)

            img_r4 = np.expand_dims(img_r4, 0)
            img_g4 = np.expand_dims(img_g4, 0)
            img_b4 = np.expand_dims(img_b4, 0)
            img_r3g = np.expand_dims(img_r3g, 0)
            img_r3b = np.expand_dims(img_r3b, 0)
            img_g3r = np.expand_dims(img_g3r, 0)
            img_g3b = np.expand_dims(img_g3b, 0)
            img_b3r = np.expand_dims(img_b3r, 0)
            img_b3g = np.expand_dims(img_b3g, 0)
            img_r2g2 = np.expand_dims(img_r2g2, 0)
            img_g2b2 = np.expand_dims(img_g2b2, 0)
            img_r2b2 = np.expand_dims(img_r2b2, 0)
            img_r2gb = np.expand_dims(img_r2gb, 0)
            img_g2rb = np.expand_dims(img_g2rb, 0)
            img_b2rg = np.expand_dims(img_b2rg, 0)

            return np.concatenate([img_r, img_g, img_b, img_r2, img_g2, img_b2, img_rg, img_gb, img_rb, img_r3, img_g3, img_b3, \
                               img_rg2, img_gb2, img_rb2, img_gr2, img_bg2, img_br2, img_rgb,\
                                img_r4, img_g4, img_b4, img_r3g, img_r3b, img_g3r, img_g3b, img_b3r, img_b3g,\
                                img_r2g2, img_g2b2, img_r2b2, img_r2gb, img_g2rb, img_b2rg])
    else:
        if num == 6:
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = np.power(img_r * img_g, 1/2)
            img_rb = np.power(img_r * img_b, 1/2)
            img_gb = np.power(img_g * img_b, 1/2)

            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)

            return np.concatenate([img_r, img_g, img_b, img_rg, img_rb, img_gb])
        elif num == 13:
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = np.power(img_r * img_g, 1/2)
            img_rb = np.power(img_r * img_b, 1/2)
            img_gb = np.power(img_g * img_b, 1/2)
            img_rg2 = np.power(img_r*img_g**2, 1/3)
            img_gb2 = np.power(img_g*img_b**2, 1/3)
            img_rb2 = np.power(img_r*img_b**2, 1/3)
            img_gr2 = np.power(img_g*img_r**2, 1/3)
            img_bg2 = np.power(img_b*img_g**2, 1/3)
            img_br2 = np.power(img_b*img_r**2, 1/3)
            img_rgb = np.power(img_r*img_g*img_b, 1/3)

            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)
            img_rg2 = np.expand_dims(img_rg2, 0)
            img_gb2 = np.expand_dims(img_gb2, 0)
            img_rb2 = np.expand_dims(img_rb2, 0)
            img_gr2 = np.expand_dims(img_gr2, 0)
            img_bg2 = np.expand_dims(img_bg2, 0)
            img_br2 = np.expand_dims(img_br2, 0)
            img_rgb = np.expand_dims(img_rgb, 0)

            return np.concatenate([img_r, img_g, img_b, img_rg, img_rb, img_gb, img_rg2, img_gb2, \
                img_rb2, img_gr2, img_bg2, img_br2, img_rgb])
        elif num == 22:
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img_rg = np.power(img_r * img_g, 1/2)
            img_rb = np.power(img_r * img_b, 1/2)
            img_gb = np.power(img_g * img_b, 1/2)
            img_rg2 = np.power(img_r*img_g**2, 1/3)
            img_gb2 = np.power(img_g*img_b**2, 1/3)
            img_rb2 = np.power(img_r*img_b**2, 1/3)
            img_gr2 = np.power(img_g*img_r**2, 1/3)
            img_bg2 = np.power(img_b*img_g**2, 1/3)
            img_br2 = np.power(img_b*img_r**2, 1/3)
            img_rgb = np.power(img_r*img_g*img_b, 1/3)
            img_r3g = np.power(img_r**3*img_g, 1/4)
            img_r3b = np.power(img_r**3*img_b, 1/4)
            img_g3r = np.power(img_g**3*img_r, 1/4)
            img_g3b = np.power(img_g**3*img_b, 1/4)
            img_b3r = np.power(img_b**3*img_r, 1/4)
            img_b3g = np.power(img_b**3*img_g, 1/4)
            img_r2gb = np.power(img_r**2*img_g*img_b, 1/4)
            img_g2rb = np.power(img_g**2*img_r*img_b, 1/4)
            img_b2rg = np.power(img_b**2*img_r*img_g, 1/4)

            # 拓展维度
            img_r = np.expand_dims(img_r, 0)
            img_g = np.expand_dims(img_g, 0)
            img_b = np.expand_dims(img_b, 0)
            img_rg = np.expand_dims(img_rg, 0)
            img_rb = np.expand_dims(img_rb, 0)
            img_gb = np.expand_dims(img_gb, 0)
            img_rg2 = np.expand_dims(img_rg2, 0)
            img_gb2 = np.expand_dims(img_gb2, 0)
            img_rb2 = np.expand_dims(img_rb2, 0)
            img_gr2 = np.expand_dims(img_gr2, 0)
            img_bg2 = np.expand_dims(img_bg2, 0)
            img_br2 = np.expand_dims(img_br2, 0)
            img_rgb = np.expand_dims(img_rgb, 0)
            img_r3g = np.expand_dims(img_r3g, 0)
            img_r3b = np.expand_dims(img_r3b, 0)
            img_g3r = np.expand_dims(img_g3r, 0)
            img_g3b = np.expand_dims(img_g3b, 0)
            img_b3r = np.expand_dims(img_b3r, 0)
            img_b3g = np.expand_dims(img_b3g, 0)
            img_r2gb = np.expand_dims(img_r2gb, 0)
            img_g2rb = np.expand_dims(img_g2rb, 0)
            img_b2rg = np.expand_dims(img_b2rg, 0)

            return np.concatenate([img_r, img_g, img_b, img_rg, img_rb, img_gb, img_rg2, img_gb2, \
                img_rb2, img_gr2, img_bg2, img_br2, img_rgb, img_r3g, img_r3b, \
                img_g3r, img_g3b, img_b3r, img_b3g, img_r2gb, img_g2rb, img_b2rg])

def define_scheduler(optimizer, hyper):
    """
    create scheduler for optimizers
    parameters:
        optimizer: the optimizer of the network
        hyper: hyper parameter dictionary
    """
    if hyper['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, hyper['decay_step'], gamma=0.5)
    elif hyper['lr_policy'] == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, hyper['mile_stones'], gamma=0.5)
    elif hyper['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_gamma = 1.0 - max(0, epoch - (hyper['epochs'] - hyper['epoch_decay'])) / hyper['epoch_decay']
            return lr_gamma
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        raise NotImplementedError('scheduler %s is not implemented' % {hyper['lr_policy']})
    return scheduler

# 存储模型函数
def save_model(model, opti, sche, hyper, global_step, epoch, best = False):
    base_name = os.path.join(hyper['check_dir'], 'model')

    if best:
        base_name += '_best'
    else:
        base_name += '_latest'
    
    check = {}
    check['model'] = model.state_dict()
    check['optimizer'] = opti.state_dict()
    check['scheduler'] = sche.state_dict()
    check['global_step'] = global_step
    check['epoch'] = epoch
    check['best_vali_loss'] = hyper['best_vali_loss']
    torch.save(check, base_name + '.pth') # 存储节点数据