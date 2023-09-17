import argparse
import os
import time
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.utils

import matplotlib

import numpy as np

from datasets.ShapeNet import ShapeNet_paired_dataset
from datasets.Pascal3D import Pascal3D_dataset
from datasets.toy import ico_paired_dataset
import models.nets
import models.loss

from utils import *

PRINT_INTERVAL = 100


def get_dataset(params):
    category = params['data']['cat']

    sets = []

    for s in ['train', 'val', 'test']:
        conf = params['data'][s]
        if conf['origin'] == 'shapenet':
            dset = ShapeNet_paired_dataset( category=category,
                                            split=conf['split'],
                                            src_dir=conf['path'],
                                            bg=conf['bg'],
                                            bg_path=conf['bg_path'],
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(size=(64, 64)),
                                                torchvision.transforms.ToTensor()
                                                ])
                                            )
        elif conf['origin'] == 'pascal': 
            dset = Pascal3D_dataset(category=category,
                                    src_dir=conf['path'],
                                    subset=conf['subset'],
                                    split=conf['split'],
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.Resize(size=(64, 64)),
                                        torchvision.transforms.ToTensor()
                                        ])
                                    )
    
        sets.append(dset)
    train_dataset, val_dataset, test_dataset = sets


    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=params['training']['bs'], shuffle=True, drop_last=True,
                        pin_memory=CUDA, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=params['training']['bs'], shuffle=False, drop_last=False,
                        pin_memory=CUDA, num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=params['training']['bs'], shuffle=False, drop_last=False,
                        pin_memory=CUDA, num_workers=16)

    return train_loader, val_loader, test_loader






def epoch(e, data, model, criterion, optimizer=None, logger=None, split=None, align_mat=None, cycle=False):

    if not split in ["train", "val", "test"]:
        print("Invalid argument set : must be train, val or test, but got ", split)
        raise SystemExit

    istrain = (split=="train")
    torch.autograd.set_grad_enabled(istrain)

    model = model.train() if istrain else model.eval()

    avg_loss_pose = AverageMeter()
    avg_loss_rec = AverageMeter()
    avg_loss_mvc = AverageMeter()
    avg_loss_sep = AverageMeter()
    avg_loss_eq = AverageMeter()
    avg_loss_sym = AverageMeter()
    avg_loss_adv = AverageMeter()
    avg_batch_time = AverageMeter()

    avg_acc = AverageMeter()
    med_err = MedianMeter()

    for i, inp in enumerate(data):
        tic = time.time()
        if len(inp) == 5:
            p_inp, c_inp, target, azimuth, elevation = inp
        else:
            p_inp, azimuth, elevation = inp
            c_inp = target = p_inp

        inp = c_inp
        bsize = inp.size(0)
        halfsize = bsize//2

        xsize = inp.size(-1)
        ysize = inp.size(-2)

        if CUDA: 
            p_inp = p_inp.float().cuda()
            c_inp = c_inp.float().cuda()
            target = target.float().cuda()
            azimuth = azimuth.float().cuda()
            elevation = elevation.float().cuda()
        loss = 0

        outputs, segmasks, student, poses, latent_code = model.forward(p_inp, c_inp, istrain)

        gt_angles = torch.stack([azimuth, elevation], dim=1)
        gt_pose = theta2sphere(gt_angles.float()/180*np.pi)


        if istrain:
            diff = (outputs - target.unsqueeze(0)).abs().mean(-1).mean(-1).mean(-1)
            _, inds = diff.min(dim=0)
            output = torch.stack([outputs[head, k] for k, head in enumerate(inds)])
            segmask = torch.stack([segmasks[head, k] for k, head in enumerate(inds)])
            pose = torch.stack([poses[k, :, head] for k, head in enumerate(inds)])
            loss += 1e0 * nn.functional.cross_entropy(student, inds)
        else:
            output = outputs[0]
            segmask = segmasks[0]
            pose = poses

        rec_loss = 1 * criterion(output, target)
        loss += 1e0 * rec_loss


        rand_az, rand_el = torch.rand_like(azimuth)*2*np.pi - np.pi, torch.rand_like(elevation)*np.pi*.6 - np.pi*.2
        rand_pose = theta2sphere(torch.stack([rand_az, rand_el], dim=1))
        rand_trans = torch.randn_like(rand_pose) * .1
        rand_focal = torch.randn_like(rand_pose[:,:1])*.1 +.25
        rand_scale = torch.randn_like(rand_pose[:,:1])*.1 + 1

        if cycle:
            rand_dec, _ =  model.decoder(latent_code, rand_pose.unsqueeze(-1), )#rand_trans, rand_focal, rand_scale)
            rand_dec = rand_dec[0]
            re_rand_dec = nn.functional.interpolate(rand_dec, inp.size(-1)).detach()
            re_rand_pose = model.pose_net(re_rand_dec).squeeze()

            probs, re_rand_pose = re_rand_pose[:,:3], re_rand_pose[:,3:]
            re_rand_pose = re_rand_pose.view(bsize,3,-1)
            norms = re_rand_pose.norm(dim=-2)
            re_rand_pose = re_rand_pose/norms.unsqueeze(-2)
            diff = (re_rand_pose - rand_pose.unsqueeze(-1)).pow(2).mean(1)
            _, inds = probs.max(dim=1)
            re_rand_pose = torch.stack([re_rand_pose[k, :, head] for k, head in enumerate(inds)])
            _, gt_inds = diff.min(dim=-1)
            loss += 1e0 * nn.functional.cross_entropy(probs, gt_inds)

            loss += 1e0*nn.functional.mse_loss(re_rand_pose, rand_pose)

        if istrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        batch_time = time.time() - tic

        if align_mat is not None:
            pose = torch.bmm(align_mat.repeat(bsize,1,1), pose.unsqueeze(-1)).view(bsize,-1)
            pred_rotmats = lookat_camera_rotation(pose)
            gt_pose = theta2sphere(torch.stack([azimuth, elevation], dim=1).float()/180*np.pi)
            gt_rotmats = lookat_camera_rotation(gt_pose)
            avg_acc.update(angle_accuracy(pred_rotmats, gt_rotmats).cpu().item())
            med_err.update(rel_angle(pred_rotmats, gt_rotmats).cpu().detach())
        if logger:
            logger.log_code("posex", pose[:,0])
            logger.log_code("posey", pose[:,1])
            logger.log_code("posez", pose[:,2])
            logger.log_code("az", azimuth)
            logger.log_code("el", elevation)

        avg_loss_rec.update(rec_loss.cpu().item())
        if i>0:
            avg_batch_time.update(batch_time)





        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   split, i,
                   len(data), batch_time=avg_batch_time, loss=avg_loss_rec))
            ""
            save_reconstruct(model.name()+"_"+str(epsilon),
                            [target[0], p_inp[0], c_inp[0], segmask[0], output[0]],
                            )
            ""

    if logger:
        logger.print_code(split)
        logger.reset_codes()
        loss_plot.update(split+" reconstruction", avg_loss_rec.avg)
        acc_plot.update(split+" accuracy", avg_acc.avg)
        loss_plot.plot()
        acc_plot.plot()
    print('===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\n\n'
          'Avg acc {acc.avg:.4f}\t'
          'Med err {err.val:.4f}\n\n'.format(
           batch_time=int(avg_batch_time.sum), loss=avg_loss_rec, acc=avg_acc, err=med_err))



    return avg_loss_pose, avg_acc.avg, med_err.val.item()



def load_partial_state_dict(model, pretrained_dict):

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.load_state_dict(pretrained_dict)


def train(params):

    early_stopping_epochs = 30

    start = params['training']['resume']
    cat = params['data']['cat']

    model = models.nets.ViewNet(cont_size=params['model']['cont_size'], n_heads=params['model']['n_heads'])

    global loss_plot
    loss_plot = Plot('loss_'+model.name()+"_"+str(epsilon), "Loss")
    global acc_plot
    acc_plot = Plot('accuracy_'+model.name()+"_"+str(epsilon), "Accuracy")

    criterion = models.loss.PerceptualLoss(losstype='both')
    optimizer = torch.optim.Adam([{'params': model.parameters()},], params['training']['lr'],)

    align_mat = torch.eye(3)

    if CUDA: 
        model = model.cuda()
        criterion = criterion.cuda()
        align_mat = align_mat.cuda()
    global pred_print

    train_dataset, val_dataset, test_dataset = get_dataset(params)


    best_val_acc = 0
    best_ep = 0

    if start != 0:
        if start == -1:
            model_id = 'best_mod.pth'
        else:
            model_id = str(start) + '.pth'
        load_path = os.path.join(params['training']['resume_path'], model_id)
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)

        align_mat = align_preds(val_dataset, model, CUDA)
        val_loss, val_acc, val_err = epoch(0, val_dataset, model, criterion, optimizer, None, split="val", align_mat=align_mat)
        test_loss, test_acc, test_err = epoch(0, test_dataset, model, criterion, optimizer, None, split="test", align_mat=align_mat)
        best_val_acc = val_acc
        print('Test acc: ', test_acc)
        print('Test err: ', test_err)
        exit()

    logger = code_logger(model.name()+"_"+str(epsilon))
    mod_logger = Model_logger(model, model.name()+"_"+cat+"_"+str(epsilon), path=params['training']['ckpt_path'])

    for i in range(start, params['training']['epochs']):
        print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")
        loss, acc, err= epoch(i, train_dataset, model, criterion, optimizer, logger, split="train", align_mat=align_mat, cycle=params['training']['cycle'])
        align_mat = align_preds(val_dataset, model, CUDA)
        val_loss, val_acc, val_err = epoch(i, val_dataset, model, criterion, optimizer, logger, split="val", align_mat=align_mat)
        create_views(val_dataset, model, CUDA, epsilon)

        if (i+1)%5==0:
            mod_logger.dump(str(i+1))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ep = i
            mod_logger.dump('best_mod')

        if i>=best_ep+early_stopping_epochs:
            print('Early stopping triggered')
            break
        else:
            print('epochs before early stopping:', early_stopping_epochs - i + best_ep)


    #Loading chkpt
    load_path = os.path.join(mod_logger.get_path(), 'best_mod.pth')
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    
    align_mat = align_preds(val_dataset, model, CUDA)
    test_loss, test_acc, test_err = epoch(i, test_dataset, model, criterion, optimizer, logger, split="test", align_mat=align_mat)

    print('Best epoch: ', best_ep)
    print('Accuracy: ', test_acc)
    print('Angular error: ', test_err)

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', dest='conf_path', required=True, help='configuration file path')
    args = parser.parse_args()

    with open(args.conf_path, 'r') as conf_file:
        cfg = yaml.safe_load(conf_file)

    global CUDA
    CUDA = cfg['training']['cuda']
    global epsilon
    epsilon = cfg['training']['eps']

    if CUDA and not torch.cuda.is_available():
        print("Warning: CUDA requested, but it is not available.")

    train(cfg)
