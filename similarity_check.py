import argparse
import logging
import os
import sys

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch_cka import CKA
from torchvision import transforms

from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils import ramps


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Pancreas_CT', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='Co_SegNet_adv_v2_lambda_1.0_con_1.0_tm_0.2', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=3, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=12, help='trained samples')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--lamda', type=float, default=1.0, help='weight to balance all losses')
parser.add_argument('--mu', type=float, default=0.01, help='weight to balance generator adversarial loss')
args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = '../data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = '../data/Pancreas'
    args.max_samples = 62
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model_1 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes - 1, mode="train")
    save_mode_path_1 = os.path.join(snapshot_path, 'best_model_1.pth'.format(args.model))
    model_1.load_state_dict(torch.load(save_mode_path_1), strict=False)
    print("init weight from {}".format(save_mode_path_1))

    model_2 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes - 1, mode="train")
    save_mode_path_2 = os.path.join(snapshot_path, 'best_model_2.pth'.format(args.model))
    model_2.load_state_dict(torch.load(save_mode_path_2), strict=False)
    print("init weight from {}".format(save_mode_path_2))
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='test',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                            split='test',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))

    trainloader = DataLoader(db_train, num_workers=2, pin_memory=True)

    layers = ["encoder.block_one.conv.1", "encoder.block_one_dw.conv.1", "encoder.block_two.conv.1",
              "encoder.block_two_dw.conv.1", "encoder.block_three.conv.1", "encoder.block_three_dw.conv.1",
              "encoder.block_four.conv.1", "encoder.block_four_dw.conv.1", "decoder1.block_five.conv.1",
              "decoder1.block_five_up.conv.1", "decoder1.block_six.conv.1", "decoder1.block_six_up.conv.1",
              "decoder1.block_seven.conv.1", "decoder1.block_seven_up.conv.1", "decoder1.block_eight.conv.1",
              "decoder1.block_eight_up.conv.1",
              "decoder1.block_nine.conv.1", "decoder1.out_conv"]

    cka = CKA(model_1, model_2,
              model1_name="VNet1", model2_name="VNet1",
              model1_layers=layers,  # List of layers to extract features from
              model2_layers=layers,
              device='cuda')

    cka.compare(trainloader)

    cka.plot_results(save_path="../assets/resnet-resnet_compare.png")
    print(cka)
    print("\n========================================")
