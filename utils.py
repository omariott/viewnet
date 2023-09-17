import matplotlib.pyplot as plt
import itertools
from PIL import Image
from skimage import io
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import colorsys
import imageio

def classification_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def rel_angle(output, target):
    n_dim = len(target.shape)
    in_dim = target.size(1)
    if n_dim == 3:
        angle = angle_rotmat(output, target)
    elif in_dim==4:
        angle = angle_quat(output, target)
    elif in_dim==3:
        angle = angle_vec(output, target)
    else:
        print("Error, target should contain 3 or 4 channels but got {}".format(in_dim))
        exit()
    return angle


def angle_rotmat(A,B):
    prod = A.bmm(B.transpose(-1,-2))
    antisim = .5 *(prod - prod.transpose(-1,-2))
    norm = torch.stack([a.trace() for a in antisim.bmm(antisim)], dim=0).mul(-.5).pow(.5)
    norm = norm.unsqueeze(-1).unsqueeze(-1)
    log = norm.asin()/norm * antisim
    dist = log.norm('fro', dim=(-2, -1))/np.sqrt(2)
    return dist/np.pi*180


def angle_quat(output, target):
    angle = (2*(output*target).sum(-1).pow(2)-1).acos()
    return angle.squeeze()/np.pi*180


def angle_vec(output, target):
    angle = (output*target).sum(-1).acos().float()
    return angle.squeeze()/np.pi*180


def angle_accuracy(output, target):
    bsize = target.size(0)
    angle = rel_angle(output, target)
    correct = angle < 30
    res = correct.float().sum()/bsize
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MedianMeter(object):
    """Computes and stores the Median and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.data = []

    def update(self, vals):
        self.data.append(vals)
        self.val = torch.cat(self.data).median()


class Plot(object):
    def __init__(self, plot_name, ylabel, path="plots"):
        self.plot_name = plot_name
        self.ylabel = ylabel
        self.path = path
        self.losses = dict()
        self.fig = plt.figure()

    def update(self, key, loss):
        if key in self.losses:
            self.losses[key].append(loss)
        else:
            self.losses[key] = [loss]

    def plot(self):
        plt.figure(self.fig.number)
        plt.clf()
        for key in self.losses:
            plt.plot(np.array(self.losses[key]), label=key)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(self.ylabel)
        plt.savefig(os.path.join(self.path, self.plot_name+'.png'))



class code_logger(object):
    def __init__(self, name, path="logs"):
        super(code_logger, self).__init__()
        if not os.path.isdir(path):
            os.mkdir(path)
        self.name = name
        self.path = path
        self.codes = dict()

    def log_code(self, key, code):
        if key in self.codes:
            self.codes[key].append(code.detach().cpu().numpy())
        else:
            self.codes[key] = [code.detach().cpu().numpy()]

    def print_code(self, plotname=""):
        if self.codes:
            if plotname!="":
                plotname = "_"+plotname

            full_code = np.stack([np.concatenate(vals) for vals in self.codes.values()], axis=-1)
            np.save(os.path.join(self.path, self.name+plotname+"_codelog"), full_code)


    def reset_codes(self):
        self.codes = dict()



def theta2sphere(theta):
    azi, elev = theta[:,0], theta[:,1]
    x = elev.cos()*azi.sin()
    y = elev.cos()*azi.cos()
    z = elev.sin()
    code = torch.stack([x, y, z], dim=1)
    return code

def sphere2theta(points):
    x, y, z = points[:,0], points[:,1], points[:,2]
    az = torch.atan2(x,y)
    elev = torch.asin(z)
    code = torch.stack([az, elev], dim=1)
    return(code)

def az2quat(azimuth):
    zeros = torch.zeros_like(azimuth)
    quat = torch.stack([(azimuth/2).cos(), zeros, zeros, (azimuth/2).sin()], dim = -1)
    return quat

def el2quat(elevation):
    zeros = torch.zeros_like(elevation)
    quat = torch.stack([(elevation/2).cos(), zeros, (elevation/2).sin(), zeros], dim = -1)
    return quat

def angle2quat(angles):
    azimuth, elevation = angles.unbind(dim=1)
    return quatprod(az2quat(azimuth), el2quat(elevation))

def quatprod(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=1)
    w2, x2, y2, z2 = q2.unbind(dim=1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    quat = torch.stack([w,x,y,z], dim=1)
    return quat

def conj(q):
    conjugator = -1*torch.ones_like(q)
    conjugator[:,0] = 1
    return q*conjugator

def quatrot(v, q):
    while(len(q.shape)<len(v.shape)):
        q = q.unsqueeze(-1)
    zeros = torch.zeros_like(v[:,0:1])
    extended_v = torch.cat([zeros,v], dim=1)
    return quatprod(quatprod(q, extended_v), conj(q))[:,1:]

def quat2rotmat(q):
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    rotmat = torch.stack(
        [
            torch.stack([1-2*y.pow(2)-2*z.pow(2), 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
            torch.stack([2*x*y + 2*z*w, 1-2*x.pow(2)-2*z.pow(2), 2*y*z - 2*x*w], dim=-1),
            torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1-2*x.pow(2)-2*y.pow(2)], dim=-1),
        ]
    , dim=-1)
    return rotmat


def angles2rotmat(az, el, ti):
    zeros = torch.zeros_like(az)
    ones = torch.ones_like(az)
    Mx = torch.stack([
        torch.stack([az.cos(), az.sin(), zeros], dim=1),
        torch.stack([-az.sin(), az.cos(), zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1),
    ], dim=1)
    My = torch.stack([
        torch.stack([el.cos(), zeros, -el.sin()], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([el.sin(), zeros, el.cos()], dim=1),
    ], dim=1)
    Mz = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, ti.cos(), ti.sin()], dim=1),
        torch.stack([zeros, -ti.sin(), ti.cos()], dim=1),
    ], dim=1)
    return Mz.bmm(My.bmm(Mx))


def logm_rots(R):
    tr = torch.stack([r.trace() for r in R])
    th = ((tr - 1) * .5).acos().unsqueeze(-1).unsqueeze(-1)
    logR = th * .5 / th.sin() * (R - R.transpose(-1, -2))
    return logR
    

def lookat_camera_rotation(camera_position, up=None, target=None,):
    """
        Recover camera rotation from position and looking point
    """
    if target is None:
        target = torch.zeros_like(camera_position)
    if up is None:    
        up = torch.zeros_like(camera_position)
        up[:,-1] += 1
    z = camera_position-target
    z = z/z.norm(dim=-1, keepdim=True)
    x = torch.cross(up,z, dim=-1)
    x = x/x.norm(dim=-1, keepdim=True)
    y = torch.cross(z,x, dim=-1)
    rotmat = torch.stack((x,y,z), dim=-1)
    return rotmat



def normalize(code, n_dim=None, pow=2):
    code_shape = code.shape
    if n_dim is not None:
        reshaped = code.view(code_shape[0], n_dim, -1)
        norm = reshaped.norm(p=pow, dim=1, keepdim=True)
        normed = reshaped/norm
    else:
        norm = code.norm(p=pow, dim=1, keepdim=True)
        normed = code/norm
    return normed.view_as(code)


def kpts2rot(kpts_a, kpts_b, noise=0.0):
    bsize = kpts_a.size(0)
    n_dim = kpts_a.size(-1)
    kpts_a = kpts_a + torch.randn_like(kpts_a)*noise
    kpts_b = kpts_b + torch.randn_like(kpts_b)*noise
    prod = kpts_a.transpose(-1,-2).bmm(kpts_b)
    u, _, vT = torch.svd(prod)
    det = torch.det(u.bmm(vT.transpose(-1,-2)))
    corrector = torch.eye(n_dim, device=vT.device).repeat(bsize,1,1)
    corrector[:,-1,-1] = det
    vT = torch.bmm(corrector.detach(), vT)
    rotmat = torch.bmm(u, vT.transpose(-1,-2))
    return rotmat

def save_reconstruct(name, images, kpts=None, diff_inds=(0,-1), path="plots", diff=True):
    if not os.path.isdir(path):
        os.mkdir(path)
    ims = []
    for i, image in enumerate(images):
        pic = image.detach().cpu().numpy()
        if kpts is not None and i<len(kpts):
            pic = add_kpts_img(pic, kpts[i])
        ims.append(pic)
    if diff:
        diff = (images[diff_inds[0]]-images[diff_inds[1]]).detach().pow(2).cpu().numpy()
        ims.append(diff/diff.max())
    out = np.concatenate(ims, axis=2).transpose((1,2,0))
    if (out<0).any() or (out>1).any():
        np.clip(out,0,1,out)
    out = (out*255).astype('uint8')
    io.imsave(os.path.join(path, name+"_reconstruct.png"), out)

class Model_logger(object):
    def __init__(self, model, modname, path):
        self.model = model
        self.path = os.path.join(path, modname)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def get_path(self):
        return self.path

    def dump(self, name):
        path = os.path.join(self.path, name)+".pth"
        torch.save(self.model.state_dict(), path)

def save_tensor_as_heatmap(tensor, name="tensor", colorbar=True):

    assert len(tensor.shape)==2

    plt.clf()
    array = tensor.detach().cpu().numpy()
    plt.imshow(array, cmap='viridis', interpolation='nearest')
    if colorbar:
        plt.colorbar()
    plt.savefig(name+'.png')

def rand_derangment(size):
    original = torch.arange(size)
    indices = torch.randperm(size)
    while ((original-indices)==0).any():
        indices = torch.randperm(size)
    return indices



def cube_like(tensor):
    bsize, n_chan, ydim, xdim, zdim = tensor.shape
    line = torch.linspace(-1,1,xdim, device=tensor.device)
    ones = torch.ones_like(line)

    ycube = line.unsqueeze(-1).unsqueeze(-1) * ones.unsqueeze(0).unsqueeze(-1) * ones.unsqueeze(0).unsqueeze(0)
    xcube = line.unsqueeze(0).unsqueeze(-1) * ones.unsqueeze(-1).unsqueeze(-1) * ones.unsqueeze(0).unsqueeze(0)
    zcube = line.unsqueeze(0).unsqueeze(0) * ones.unsqueeze(0).unsqueeze(-1) * ones.unsqueeze(-1).unsqueeze(-1)

    cube = torch.stack([ycube,xcube,zcube], dim=-1)

    return cube



def save_gif(batch, name, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    seq = batch.unbind(0)
    images = []
    for i, image in enumerate(seq):
        im = image.detach().cpu().numpy().transpose((1,2,0))
        im = (im * 255).round().astype(np.uint8)
        images.append(im)
    imageio.mimsave(os.path.join(path, name+".gif"), images)


def create_views(data, model, CUDA, epsilon):

    torch.autograd.set_grad_enabled(False)
    model.eval()

    for i, inp in enumerate(data):
        bsize = 36
        p_inp, c_inp, *rest = inp
        inp = p_inp
        
        inp = nn.functional.interpolate(inp, 64)

        if CUDA: 
            inp = inp.float().cuda()
            c_inp = c_inp.float().cuda()

        elevation = torch.ones(bsize, device = inp.device) * 20
        azimuth = torch.linspace(0, 350, bsize, device = inp.device)

        c_pose = torch.stack([azimuth, elevation], dim=-1)
        c_pose = theta2sphere(c_pose.float()/180*np.pi)
        tar_rot = c_pose
        latent_code = model.code_net(c_inp)[0].squeeze().repeat(bsize,1)
        outputs, segmask = model.decoder(latent_code, c_pose.unsqueeze(-1))
        output = outputs[0]
        save_gif(output, model.name()+"_"+str(epsilon), "rotplots")
        break




def step_drawing(data, model, CUDA, epsilon):

    torch.autograd.set_grad_enabled(False)


    model.eval()

    for i, inp in enumerate(data):
        bsize = 36
        p_inp, c_inp, *rest = inp
        inp = p_inp
        
        inp = nn.functional.interpolate(inp, 64)

        if CUDA:
            inp = inp.float().cuda()
            c_inp = c_inp.float().cuda()
            
        torchvision.utils.save_image(inp[0],  'draw/inp.png')
        torchvision.utils.save_image(c_inp[0],  'draw/c_inp.png')
        inp = inp[:,:3]
        c_inp = c_inp[:,:3]
        
        
        poses = model.pose_net(inp).squeeze()
        student, poses = poses[:,:3], poses[:,3:]
        poses = poses.view(-1,3,3)
        _, inds = student.max(dim=1)
        pose = torch.stack([poses[k, :, head] for k, head in enumerate(inds)])
        p_elev = sphere2theta(pose)[0,1].item()
        
        latent_code = model.code_net(c_inp)[0].squeeze().repeat(bsize,1)

        elevation = torch.ones(bsize, device = inp.device) * 20
        azimuth = torch.linspace(0, 350, bsize, device = inp.device)
        focal = torch.zeros(bsize, device = inp.device) 
        c_pose = torch.stack([azimuth, elevation], dim=-1)
        c_pose = theta2sphere(c_pose.float()/180*np.pi)
        outputs, segmask = model.decoder(latent_code, c_pose.unsqueeze(-1), focal=focal.unsqueeze(-1))
        output = torch.cat([outputs[0], segmask[0].mean(dim=1, keepdim=True)], dim=1)
        for i, im in enumerate(output):
            torchvision.utils.save_image(im,  'draw/'+str(i)+'_ortho.png')
        
        
        outputs, segmask = model.decoder(latent_code, c_pose.unsqueeze(-1), focal=focal.unsqueeze(-1))
        output = torch.cat([outputs[0], segmask[0].mean(dim=1, keepdim=True)], dim=1)
        for i, im in enumerate(output):
            torchvision.utils.save_image(im,  'draw/'+str(i)+'_orthorot.png')
            
            
            
        outputs, segmask = model.decoder(latent_code, c_pose.unsqueeze(-1),  drawing_pose=pose[0], focal=focal.unsqueeze(-1))
        output = torch.cat([outputs[0], segmask[0].mean(dim=1, keepdim=True)], dim=1)
        for i, im in enumerate(output):
            torchvision.utils.save_image(im,  'draw/'+str(i)+'.png')
            
            
        out, seg = model.decoder(latent_code[0:1], pose[0:1].unsqueeze(-1))
        out = torch.cat([out[0], seg[0].mean(dim=1, keepdim=True)], dim=1)
        torchvision.utils.save_image(out,  'draw/out.png')
            
        break

def print_batch(data, model, CUDA, epsilon):
    torch.autograd.set_grad_enabled(False)
    model.eval()

    for i, inp in enumerate(data):
        p_inp, c_inp, *rest = inp
        inp = p_inp
        bsize = p_inp.size(0)
        

        if CUDA: 
            inp = inp.float().cuda()
            c_inp = c_inp.float().cuda()
            
        p_inp = inp[:,:3]
        c_inp = c_inp[:,:3]
        
        
        poses = model.pose_net(p_inp).squeeze()
        student, poses = poses[:,:model.n_heads], poses[:,model.n_heads:]
        poses = poses.view(-1,3,-1)
        _, inds = student.max(dim=1)
        pose = torch.stack([poses[k, :, head] for k, head in enumerate(inds)])
        
        latent_code = model.code_net(c_inp).squeeze()
        outputs, segmask = model.decoder(latent_code, pose.unsqueeze(-1))
        output = torch.cat([outputs[0], segmask[0].mean(dim=1, keepdim=True)], dim=1)
        for i, im in enumerate(output):
            torchvision.utils.save_image(torch.cat([inp[i], im], dim=-1),  'batch/'+str(i)+'.png')
        

        break


def create_grid(data, model, CUDA, epsilon):


    torch.autograd.set_grad_enabled(False)
    model.eval()

    for i, inp in enumerate(data):
        bsize = 10
        p_inp, c_inp, *rest = inp
        inp = p_inp
        
        inp = nn.functional.interpolate(inp, 64)

        if CUDA: 
            inp = inp.float().cuda()
            c_inp = c_inp.float().cuda()
        cont_inp = c_inp[:,:3]

        elevation =  torch.linspace(-20, 40, bsize, device = inp.device) 
        azimuth = torch.ones(bsize, device = inp.device) * -90

        c_pose = torch.stack([azimuth, elevation], dim=-1)
        c_pose = theta2sphere(c_pose.float()/180*np.pi)
        tar_rot = c_pose
        latent_code = model.code_net(cont_inp)[0].squeeze().repeat(bsize,1)
        outputs, segmask = model.decoder(latent_code, c_pose.unsqueeze(-1))
        output = torch.cat([outputs[0], segmask[0].mean(dim=1, keepdim=True)], dim=1)
        stripe = torch.cat([c_inp[0:1], output], dim=0)
        torchvision.utils.save_image(stripe, 'grid/grid.png', nrow=11)
        break


def align_preds(data, model, CUDA, samples=64):
    torch.autograd.set_grad_enabled(False)
    model.eval()
    sample_count = 0
    preds = []
    gt = []
    for i, inp in enumerate(data):
        pose_inp, _, _, azimuth, elevation = inp
        bsize = pose_inp.size(0)
        sample_count += bsize
        if CUDA:
            pose_inp = pose_inp.float().cuda()
            azimuth = azimuth.cuda()
            elevation = elevation.cuda()
        gt_angles = torch.stack([azimuth, elevation], dim=1)
        gt_pose = theta2sphere(gt_angles.float()/180*np.pi)
        
        pose_inp = nn.functional.interpolate(pose_inp, 64)
        
        pose = model.pose_net(pose_inp).view(bsize, -1)
        probs, pose = pose[:,:model.n_heads], pose[:,model.n_heads:]
        pose = pose.view(bsize,3,-1)
        norms = pose.norm(dim=-2)
        pose = pose/norms.unsqueeze(-2)
        _, inds = probs.max(dim=1)
        pred_pose = torch.stack([pose[k, :, head] for k, head in enumerate(inds)])
        preds.append(pred_pose)
        gt.append(gt_pose)
        if sample_count>samples:
            break
    
    preds = torch.cat(preds, dim=0)
    gt = torch.cat(gt, dim=0)
    rotmat =  kpts2rot(gt.unsqueeze(0), preds.unsqueeze(0))
    return rotmat