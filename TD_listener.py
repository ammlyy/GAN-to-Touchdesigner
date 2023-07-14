# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import click
import dnnlib
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm

import legacy

#----------------------------------------------------------------------------
import sys

from viz.renderer import CaptureSuccess
sys.path.append('Library')

import numpy as np
import argparse
import time
import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GLU import *

def run_synthesis_net(net, *args, capture_layer=None, **kwargs): # => out, layers
    submodule_names = {mod: name for name, mod in net.named_modules()}
    unique_names = set()
    layers = []

    def module_hook(module, _inputs, outputs):
        outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
        outputs = [out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]]
        for idx, out in enumerate(outputs):
            if out.ndim == 5: # G-CNN => remove group dimension.
                out = out.mean(2)
            name = submodule_names[module]
            if name == '':
                name = 'output'
            if len(outputs) > 1:
                name += f':{idx}'
            if name in unique_names:
                suffix = 2
                while f'{name}_{suffix}' in unique_names:
                    suffix += 1
                name += f'_{suffix}'
            unique_names.add(name)
            shape = [int(x) for x in out.shape]
            dtype = str(out.dtype).split('.')[-1]
            layers.append(dnnlib.EasyDict(name=name, shape=shape, dtype=dtype))
            if name == capture_layer:
                raise CaptureSuccess(out)

    hooks = [module.register_forward_hook(module_hook) for module in net.modules()]
    try:
        out = net(*args, **kwargs)
    except CaptureSuccess as e:
        out = e.out
    for hook in hooks:
        hook.remove()
    return out, layers

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
def generate(network_pkl):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to('cuda') # type: ignore

    # window details
    width = 256
    height = 256
    display = (width,height)
    
    senderName = 'GAN out'
    silent = False
    
    # window setup
    pygame.init() 
    pygame.display.set_caption(senderName)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # init spout sender
    spoutSender = SpoutSDK.SpoutSender()
    spoutSenderWidth = width
    spoutSenderHeight = height
    # Its signature in c++ looks like this: bool CreateSender(const char *Sendername, unsigned int width, unsigned int height, DWORD dwFormat = 0);
    spoutSender.CreateSender(senderName, spoutSenderWidth, spoutSenderHeight, 0)
    # create textures for spout receiver and spout sender 
    textureSendID = glGenTextures(1)
    frame_idx = 0
    while(True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                spoutReceiver.ReleaseReceiver()
                pygame.quit()
                quit()

         #initial seeds?
        w0_seeds = [[0, 1]]
        seed = 234
        # Generate random latents.
        all_seeds = [seed for seed, _weight in w0_seeds]
        all_seeds = list(set(all_seeds))
        all_zs = np.zeros([len(all_seeds), G.z_dim], dtype=np.float32)
        all_cs = np.zeros([len(all_seeds), G.c_dim], dtype=np.float32)
        for idx, seed in enumerate(all_seeds):
            rnd = np.random.RandomState(seed)
            all_zs[idx] = rnd.randn(G.z_dim)
            if G.c_dim > 0:
                all_cs[idx, rnd.randint(G.c_dim)] = 1

        # Run mapping network.
        w_avg = G.mapping.w_avg
        all_zs = (torch.from_numpy(all_zs))
        all_cs = (torch.from_numpy(all_cs))
        all_ws = G.mapping(z=all_zs, c=all_cs, truncation_psi=1.0, truncation_cutoff=0) - w_avg
        all_ws = dict(zip(all_seeds, all_ws))

        # Calculate final W.
        w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
        stylemix_idx = [idx for idx in stylemix_idx if 0 <= idx < G.num_ws]
        if len(stylemix_idx) > 0:
            w[:, stylemix_idx] = all_ws[seed][np.newaxis, stylemix_idx]
        w += w_avg

        # Run synthesis network.
        synthesis_kwargs = dnnlib.EasyDict(noise_mode='const', force_fp32=False)
        torch.manual_seed(seed)
        out, layers = run_synthesis_net(G.synthesis, w, capture_layer=layer_name, **synthesis_kwargs)


        # Select channels and compute statistics.
        out = out[0].to(torch.float32)
        if sel_channels > out.shape[0]:
            sel_channels = 1
        base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
        sel = out[base_channel : base_channel + sel_channels]

        # Scale and convert to uint8.
        img = sel
        img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)


    # setup the texture so we can load the output into it
        glBindTexture(GL_TEXTURE_2D, textureSendID);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # copy output into texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            
        # setup window to draw to screen
        glActiveTexture(GL_TEXTURE0)
        # clean start
        glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT )
        # reset drawing perspective
        glLoadIdentity()
        # draw texture on screen
        glBegin(GL_QUADS)

        glTexCoord(0,0)        
        glVertex2f(0,0)

        glTexCoord(1,0)
        glVertex2f(width,0)

        glTexCoord(1,1)
        glVertex2f(width,height)

        glTexCoord(0,1)
        glVertex2f(0,height)

        glEnd()
        
        if silent:
            pygame.display.iconify()
                
        # update window
        pygame.display.flip()        

        spoutSender.SendTexture(textureSendID.item(), GL_TEXTURE_2D, spoutSenderWidth, spoutSenderHeight, False, 0)


if __name__ == "__main__":
    generate()

