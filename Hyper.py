import torch
hyper = {
    'epochs':1000,
    'batch_size': 120, # training batch size
    'vali_batch_size': 200, # validation batch_size
    'patch_size': 256, # the size of input patch
    'save_fre': 5, # save per 5 epoch
    'vali_fre': 5, # validate per 5 epoch
    'lr': 0.0001, # learning rate
    'b1': 0.9, # Adam b1
    'b2': 0.999, # Adam b2
    'nfc': 64, # base number of feature map channel
    'dev': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), # device GPU or CPU
    'lr_policy': 'step', # 'step'/'multistep'/'linear'
    'decay_step': 200, # work when lr_policy = step, the decay factor of learning rate
    'mile_stones': [], # work when lr_policy = multistep
    'epoch_decay': 500, # work when lr_policy = linear
    'best_vali_loss': None, # the best validation loss
    'check_dir': './check_points', # path to save check points

    # 模型参数
    'img_chan': 3, # the number of input image channel
    'channel_redu': 2, # reduction rate in channel attention
    'neuron_redu': 2, # reduction rate in neuron attention
    'term': 11, # the number of polynomial terms. if root = False, it is chosen from [4, 5, 6, 7, 9, 11, 19, 34]，and 4 represents log term;  if root = True, it is chosen from [6, 13]
    'patch_num': 4, # the number of patches extracted from each image
    'linear_num': 4, # the number of linear layers
    'root': False, # using root-polynomial extension or not
    'channel_attention':False, # using channel attention module or not
    'neuron_attention': False, # using neuron attention module or not
    'residual': True # using residual term or not
}
