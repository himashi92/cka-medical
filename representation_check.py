import math

import h5py
from anatome import SimilarityHook

from torch.autograd import Variable

import os
import argparse
import torch
import numpy as np
from networks.net_factory import net_factory
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Pancreas_CT', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Co_SegNet_adv_v2_lambda_1.0_con_1.0_tm_0.2', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=6, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path ="./model/{}_{}_{}_labeled/{}".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum,
                                                                     FLAGS.model)
test_save_path = "./model/{}_{}_{}_labeled/{}_predictions_best/".format(FLAGS.dataset_name, FLAGS.exp,
                                                                                   FLAGS.labelnum, FLAGS.model)

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    FLAGS.root_path = '../data/LA'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                  image_list]

elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = '../data/Pancreas'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]


dataset = 'spleen'
perc = '30_perc'
image_root = '../data/Pancreas/Pancreas_h5/'
gt_root = '../data/Pancreas/Pancreas_h5/'


class Network(object):
    def gaussian(self, ins, mean=0, stddev=0.05):

        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        x = ins + noise
        return x

    def run(self):
        model_1 = net_factory(net_type='vnet', in_chns=1, class_num=num_classes - 1, mode="test")
        save_mode_path_1 = os.path.join(snapshot_path, 'best_model_1.pth'.format(FLAGS.model))
        model_1.load_state_dict(torch.load(save_mode_path_1), strict=False)
        print("init weight from {}".format(save_mode_path_1))
        model_1.eval()

        model_2 = net_factory(net_type='vnet', in_chns=1, class_num=num_classes - 1, mode="test")
        save_mode_path_2 = os.path.join(snapshot_path, 'best_model_2.pth'.format(FLAGS.model))
        model_2.load_state_dict(torch.load(save_mode_path_2), strict=False)
        print("init weight from {}".format(save_mode_path_2))
        model_2.eval()

        model_1.cuda()
        model_2.cuda()

        metric_detail=1
        val_loader = tqdm(image_list) if not metric_detail else image_list

        data = next(iter(val_loader))
        layers = ["encoder.block_one.conv.1", "encoder.block_one_dw.conv.1", "encoder.block_two.conv.1", "encoder.block_two_dw.conv.1", "encoder.block_three.conv.1", "encoder.block_three_dw.conv.1", "encoder.block_four.conv.1", "encoder.block_four_dw.conv.1", "encoder.block_five.conv.1",
        "decoder1.block_five_up.conv.1", "decoder1.block_six.conv.1", "decoder1.block_six_up.conv.1", "decoder1.block_seven.conv.1", "decoder1.block_seven_up.conv.1", "decoder1.block_eight.conv.1", "decoder1.block_eight_up.conv.1",
         "decoder1.block_nine.conv.1", "decoder1.out_conv"]
        layers_v1 =["encoder", "decoder1"]
        h5f = h5py.File(data, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        stride_xy = 18
        stride_z = 4
        w, h, d = image.shape

        # if the size of image is less than patch_size, then padding it
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0] - w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1] - h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2] - d
            add_pad = True
        else:
            d_pad = 0
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
        if add_pad:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                           constant_values=0)
        ww, hh, dd = image.shape

        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

        counter = 0
        for x in range(0, sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()
                    counter = counter + 1
                    if counter == 1:
                        with torch.no_grad():

                            data = test_patch
                            print(f"Number of layers {len(layers)}")
                            l = layers[17]
                            for i in layers:
                                noised_data = self.gaussian(data)
                                hook1 = SimilarityHook(model_1, l, 'lincka')
                                hook2 = SimilarityHook(model_2, i, 'lincka')
                                model_1.eval()
                                model_2.eval()
                                with torch.no_grad():
                                    model_1(noised_data)
                                    model_2(noised_data)

                                cka = hook1.distance(hook2)
                                print(cka)


if __name__ == '__main__':
    train_network = Network()
    train_network.run()
