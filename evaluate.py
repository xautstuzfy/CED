import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
import datasets
from core.ced import CED



@torch.no_grad()
def validate_Canon(model, iters=24):
    """ Perform evaluation on the Canon (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.Canon(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation Canon EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    return {'canon': epe}

@torch.no_grad()
def validate_chairs(model, iters=32):
    """ Perform evaluation on the FCDN (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    return {'chairs': epe}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='D:\\CED\\checkpoints\\FCDNed.pth',help="restore checkpoint")
    parser.add_argument('--dataset', default='chairs',help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(CED(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'canon':
            validate_Canon(model.module)


