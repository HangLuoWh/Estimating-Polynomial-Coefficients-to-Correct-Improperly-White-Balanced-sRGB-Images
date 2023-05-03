import torch
from torchvision import transforms
from Polynomial_Net import PolyNet
import os
import numpy as np
from evaluation.get_metadata import get_metadata
from evaluation.evaluate_cc import evaluate_cc
from Hyper import hyper
import utils
import cv2
from PIL import Image

def img_correct(poly_net, img, trans, dev):
    """
    image correction function
    parameter:
        poly_net: trained model
        img: input image
        trans: transforms
        dev: device
    """
    # computer polynomial coefficients using our model
    input_ten = trans(img).to(dev)
    with torch.no_grad():
        used_patch = torch.unsqueeze(input_ten, 0)
        _, img_para = poly_net(used_patch)

    # image correction
    img_para = img_para[0, :, :].view(-1, 3).detach().cpu().numpy()
    img = img.astype(np.float64)/255
    img_ext = utils.extend(img, hyper['term'], hyper['root']) # polynomially extend image
    img_cor = np.einsum('cij,cf->ijf', img_ext, img_para) # apply the polynomial coefficients to the extended image

    if hyper['residual']:
        img_cor = np.clip(img_cor + img - 0.5, 0, 1)
    else:
        img_cor = np.clip(img_cor, 0, 1)

    return (img_cor * 255).astype('uint8')

def test(poly_net, set_name):
    """
    parameter:
        poly_net: the model
        set_name: set name, ['set1_no_chart', 'set2', 'cube']
    """
    trfs = transforms.Compose([
        transforms.ToTensor()
    ])
    # configure dataset pathes
    ds_dir = './train_set'
    ds_set = os.path.join(ds_dir, set_name)
    input_dir = os.path.join(ds_set, 'input')
    gt_dir = os.path.join(ds_set, 'ground_truth')

    # load image list
    if set_name == 'set1_no_chart': # set1_no_chart needs metadata
        input_meta_dir = os.path.join(ds_set, 'input_metadata')
        vali_ls = utils.load_test_ls(ds_set) # load test image list
    else:
        vali_ls = os.listdir(input_dir)
    vali_ls.sort()
    
    # measure error
    all_error = np.zeros(shape=(len(vali_ls), 3))
    i = 0
    for img in vali_ls: # iterate on test set and compute image error
        # read input image, ground truth, metadata
        img_BGR = cv2.imread(os.path.join(input_dir, img), cv2.IMREAD_COLOR)
        img_data = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB) # BGR->RGB

        if set_name == 'set1_no_chart':
            img_meta = get_metadata(img, 'RenderedWB_Set1', input_meta_dir)
        elif set_name == 'set2':
            img_meta = get_metadata(img, 'RenderedWB_Set2')
        else:
            img_meta = get_metadata(img, 'Rendered_Cube+')
        
        gt_BGR = cv2.imread(os.path.join(gt_dir, img_meta['gt_filename']), cv2.IMREAD_COLOR) # ground truth
        img_gt = cv2.cvtColor(gt_BGR, cv2.COLOR_BGR2RGB)

        # image correction
        img_cor = img_correct(poly_net, img_data, trfs, dev)
        img_cor_pil = Image.fromarray(img_cor)
        img_cor_pil.save(f'./my_correction/{img}')
        deltaE00, MSE, MAE = evaluate_cc(img_cor, img_gt, img_meta["cc_mask_area"], opt=3)
        all_error[i, :] = [deltaE00, MSE[0], MAE]
        print(f'set:{set_name}, img:{img} done! remain:{len(vali_ls)-i-1}, error:{np.round(all_error[i, :], 2)}')
        i += 1
    np.save('test_error.npy', all_error) # save test errors to file test_error.npy

if __name__ == "__main__":
    os.makedirs('my_correction', exist_ok=True) # directory to save corrected images
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # dataset = ['set1_no_chart', 'set2', 'cube']
    dataset = ['set1_no_chart']
    #load model
    poly_net = PolyNet(hyper).to(hyper['dev']) # 多项式网络
    check_point_poly = torch.load('./check_points/model_best.pth') # 多项式网络训练好的参数
    poly_net.load_state_dict(check_point_poly['model'])
    poly_net.eval()
    poly_net.to(dev)
    # begin test
    for img_set in dataset:
        test(poly_net, img_set)