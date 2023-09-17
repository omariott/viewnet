import torch
import torch.nn as nn
import numpy as np

from torch.nn.parameter import Parameter

from  utils import *


class shape_printer(nn.Module):
    """
    Debug tool to print layer shapes
    """

    def __init__(self, num_channels=3, code_size=128):
        super(shape_printer, self).__init__()

    def forward(self, inp):
        print(inp.shape)
        return inp




class ViewNet(nn.Module):
    """
    Base autoencoder architecture
    """

    def __init__(self, in_channels=3, out_channels=3, cont_size=256, n_heads=12):
        super(ViewNet, self).__init__()
        self.cont_size = cont_size
        self.pose_net = Encoder(3, 4 * n_heads)
        self.adv_net = None
        self.code_net = Encoder(3, cont_size)
        self.decoder = VoxelDec(out_channels, cont_size)
        self.n_heads = n_heads

    def forward(self, p_inp, c_inp, istrain=False):

        bsize = p_inp.size(0)

        pose = self.pose_net(p_inp).view(bsize,-1)

        student, poses = pose[:,:self.n_heads], pose[:,self.n_heads:]
        poses = poses.view(bsize,3,-1)
        norms = poses.norm(dim=-2)
        poses = poses/norms.unsqueeze(-2)

        latent_code = self.code_net(c_inp).view(bsize,-1)

        if istrain:
            outputs, segmask = self.decoder(latent_code, poses)
        else:
            _, inds = student.max(dim=1)
            pose = torch.stack([poses[k, :, head] for k, head in enumerate(inds)])
            outputs, segmask = self.decoder(latent_code, pose.unsqueeze(-1))
            poses = pose

        return outputs, segmask, student, poses, latent_code

    def name(self):
        return "ViewNet"


class VoxelDec(nn.Module):
    """
    Base decoder architecture
    """

    def __init__(self, num_channels=3, code_size=128, logits=False):
        super(VoxelDec, self).__init__()
        width = 1
        self.conv_net = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose3d(width * 512, width * 128, 4, stride=2, padding=1),
                nn.InstanceNorm3d(width * 128),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose3d(width * 128, width * 128, 3, stride=1, padding=1),
                nn.InstanceNorm3d(width * 128),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose3d(width * 128, width * 128, 4, stride=2, padding=1),
                nn.InstanceNorm3d(width * 128),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose3d(width * 128, width * 128, 3, stride=1, padding=1),
                nn.InstanceNorm3d(width * 128),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose3d(width * 128, width * 16, 4, stride=2, padding=1),
                nn.InstanceNorm3d(width * 16),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose3d(width * 16, width * 16, 3, stride=1, padding=1),
                nn.InstanceNorm3d(width * 16),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose3d(width * 16, width * 4, 4, stride=2, padding=1),
            ),

        ])
        
        self.adaptors = nn.ModuleList([
            nn.Linear(code_size, 2 * width * 512),
            nn.Linear(code_size, 2 * width * 128),
            nn.Linear(code_size, 2 * width * 128),
            nn.Linear(code_size, 2 * width * 128),
            nn.Linear(code_size, 2 * width * 128),
            nn.Linear(code_size, 2 * width * 16),
            nn.Linear(code_size, 2 * width * 16),
        ])
        
        self.base_code = Parameter(torch.randn(1, 512, 4, 4, 4), requires_grad=True)
        self.focal_length = Parameter(torch.ones(1)*.4, requires_grad=True)

    def forward(self, inp, pose, translation=None, focal=None, mod_scale=None, code=None, flipmask=None, drawing_pose=None):
        bsize = inp.size(0)
        if code is None:
            code = self.base_code.repeat(bsize,1,1,1,1)
        output = code
        for i, conv in enumerate(self.conv_net):
            adaptor = self.adaptors[i]
            adapt_vals = adaptor(inp)
            adapt_vals = adapt_vals.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            half_chan = adapt_vals.size(1)//2
            scale, trans = adapt_vals[:,:half_chan], adapt_vals[:,half_chan:]
            output = output * scale + trans
            output = conv(output)
        canonical_out = output
        output = output.sigmoid()
        outs, masks = [], []
        if focal is None:
            focal = self.focal_length.repeat(bsize,1)
        if drawing_pose is not None:
            focal = focal*0 + 1
            output = rotate_3d_embedding(output, drawing_pose.repeat(bsize, 1), padding_mode='zeros', mode='bilinear', trans=translation, focal=focal, scale=mod_scale)
            focal = focal*0 
            output = rotate_3d_embedding(output, drawing_pose.repeat(bsize, 1), back=True, padding_mode='zeros', mode='bilinear', trans=translation, focal=focal, scale=mod_scale)
        for head in range(pose.size(-1)):
            out = rotate_3d_embedding(output, pose[..., head], padding_mode='zeros', mode='bilinear', trans=translation, focal=focal, scale=mod_scale)
            bsize, chan, size, _, _ = out.shape
            occupancy_mask, cubes = out[:,0:1], out[:,1:]
            ray_traced = cumulative_probs(occupancy_mask)
            dmap = (ray_traced).repeat(1,cubes.size(1),1,1,1)
            cubes = cubes*dmap
            out = cubes.sum(-3)
            seg_mask = dmap.sum(-3)
            outs.append(out)
            masks.append(seg_mask)
        output = torch.stack(outs)
        seg_mask = torch.stack(masks)
        return output, seg_mask

    





class Encoder(nn.Module):
    """
    Base feature_extractor architecture
    """

    def __init__(self, num_channels=3, code_size=128, logits=False):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(

            nn.ReplicationPad2d(1),
            nn.Conv2d(num_channels, 64, (3, 3), stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 128, (3, 3), stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(128, 256, (3, 3), stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(256, 512, (3, 3), stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(512, 1024, (3, 3), stride=2, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (2), stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, code_size, (1), stride=1, padding=0, dilation=1),

        )
        self.logits = logits

    def forward(self, inp):
        features = self.net(inp)
        if self.logits:
            features = features.sigmoid()
        return features

class vgg_pose(nn.Module):
    def __init__(self):
        super(vgg_pose, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features.eval()
        self.net = nn.Sequential(nn.Conv2d(512, 1024, 4),
                                 nn.ReLU(),
                                 nn.Conv2d(1024, 1024, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(1024, 3, 1),)

    def forward(self, inp):
        x = inp
        x = self.features(x)
        return self.net(x.detach())


def rotate_3d_embedding(code, pose, trans=None, focal=None, scale=None, back=False, mode='bilinear', padding_mode='zeros',):
    bsize, cube_length = code.size(0), code.size(-1)
    
    if pose.size(1)==3:
        rotmats = lookat_camera_rotation(pose)
    elif pose.size(1)==4:
        rotmats = quat2rotmat(pose)
    if back:
        rotmats = rotmats.transpose(-1, -2)
    pad = torch.zeros_like(rotmats[:,:,0:1])
    
    grids = projection_grid(rotmats, trans, focal, scale, cube_length)
    rot_code = nn.functional.grid_sample(code, grids, mode=mode, padding_mode=padding_mode, align_corners=False)

    return rot_code

def projection_grid(rotmat, trans, focal, scale, resolution):
        corners = torch.tensor([[1,1,1],
                                [1,1,-1],
                                [1,-1,1],
                                [1,-1,-1],
                                [-1,1,1],
                                [-1,1,-1],
                                [-1,-1,1],
                                [-1,-1,-1]],
                                device=rotmat.device).repeat(rotmat.size(0),1,1).float()
        #perspective distorsion
        if focal is not None:
            corners[:,[0,2,4,6],:2] *= 1 + focal.unsqueeze(1)
            corners[:,[1,3,5,7],:2] *= 1 - focal.unsqueeze(1)
        #distace scaling
        if scale is not None:
            corners *= scale.unsqueeze(1)
        #rotation
        corners = rotmat.bmm(corners.transpose(-2,-1)).transpose(-2,-1)
        #translation
        if trans is not None:
            corners += trans.unsqueeze(1)
        corners = corners.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        #grid
        line = torch.arange(.5/resolution,1,1/resolution, device=rotmat.device)
        ones = torch.ones_like(line)
        x_grid = (ones.unsqueeze(-1).unsqueeze(-1)*ones.unsqueeze(0).unsqueeze(-1)*line.unsqueeze(0).unsqueeze(0)).unsqueeze(0).unsqueeze(-1)
        y_grid = (ones.unsqueeze(-1).unsqueeze(-1)*line.unsqueeze(0).unsqueeze(-1)*ones.unsqueeze(0).unsqueeze(0)).unsqueeze(0).unsqueeze(-1)
        z_grid = (line.unsqueeze(-1).unsqueeze(-1)*ones.unsqueeze(0).unsqueeze(-1)*ones.unsqueeze(0).unsqueeze(0)).unsqueeze(0).unsqueeze(-1)
        
        sample_grid =     x_grid *     y_grid *     z_grid * corners[:,:,:,:,0] \
                    +     x_grid *     y_grid * (1-z_grid) * corners[:,:,:,:,1] \
                    +     x_grid * (1-y_grid) *     z_grid * corners[:,:,:,:,2] \
                    +     x_grid * (1-y_grid) * (1-z_grid) * corners[:,:,:,:,3] \
                    + (1-x_grid) *     y_grid *     z_grid * corners[:,:,:,:,4] \
                    + (1-x_grid) *     y_grid * (1-z_grid) * corners[:,:,:,:,5] \
                    + (1-x_grid) * (1-y_grid) *     z_grid * corners[:,:,:,:,6] \
                    + (1-x_grid) * (1-y_grid) * (1-z_grid) * corners[:,:,:,:,7]
        

        return sample_grid


def gaussian_cube(model, sigma=.05):
    dim = model.size(-1)
    line = torch.linspace(-1, 1, dim, device=model.device)
    gaussian = (-line.pow(2)/sigma).exp()
    cube = gaussian.unsqueeze(0).unsqueeze(0) * gaussian.unsqueeze(0).unsqueeze(-1) * gaussian.unsqueeze(-1).unsqueeze(-1)
    prior_cube =  (model*2-1 + cube.unsqueeze(0).unsqueeze(0)).clamp(min=0, max=1)
    return prior_cube
    

def cumulative_probs(cubes, normalized=False, gaussian_prior=True):
    if normalized:
        minvals = -nn.functional.adaptive_max_pool3d(-cubes, 1)
        maxvals = nn.functional.adaptive_max_pool3d(cubes, 1)
        cubes = (cubes - minvals)/maxvals
    if gaussian_prior:
        cubes = gaussian_cube(cubes)
    cum_probs = torch.zeros_like(cubes)
    inv_cum_prob = torch.ones_like(cubes[:,:,0])
    for i in range(cubes.size(2)):
        cur_prob = cubes[:,:,i]
        cum_probs[:,:,i] = inv_cum_prob * cur_prob
        inv_cum_prob = inv_cum_prob * (1-cur_prob)
    return cum_probs
