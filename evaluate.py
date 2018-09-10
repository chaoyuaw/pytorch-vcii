import argparse
import os
import time

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable
import torch.utils.data as data

from util import eval_forward, evaluate, get_models, set_eval, save_numpy_array_as_image
from torchvision import transforms
from dataset import get_loader


def save_codes(name, codes):
  print(codes)
  codes = (codes.astype(np.int8) + 1) // 2
  export = np.packbits(codes.reshape(-1))
  np.savez_compressed(
      name + '.codes',
      shape=codes.shape,
      codes=export)


def save_output_images(name, ex_imgs):
  for i, img in enumerate(ex_imgs):
    save_numpy_array_as_image(
      '%s_iter%02d.png' % (name, i + 1), 
      img
    )


def finish_batch(args, filenames, original, out_imgs,
                 losses, code_batch, output_suffix):

  all_losses, all_msssim, all_psnr = [], [], []
  for ex_idx, filename in enumerate(filenames):
      filename = filename.split('/')[-1]
      if args.save_codes:
        save_codes(
          os.path.join(args.out_dir, output_suffix, 'codes', filename),
          code_batch[:, ex_idx, :, :, :]
        )

      if args.save_out_img:
        save_output_images(
          os.path.join(args.out_dir, output_suffix, 'images', filename),
          out_imgs[:, ex_idx, :, :, :]
        )

      msssim, psnr = evaluate(
        original[None, ex_idx], 
        [out_img[None, ex_idx] for out_img in out_imgs])

      all_losses.append(losses)
      all_msssim.append(msssim)
      all_psnr.append(psnr)

  return all_losses, all_msssim, all_psnr


def run_eval(model, eval_loader, args, output_suffix=''):

  for sub_dir in ['codes', 'images']:
    cur_eval_dir = os.path.join(args.out_dir, output_suffix, sub_dir)
    if not os.path.exists(cur_eval_dir):
      print("Creating directory %s." % cur_eval_dir)
      os.makedirs(cur_eval_dir)

  all_losses, all_msssim, all_psnr = [], [], []

  start_time = time.time()
  for i, (batch, ctx_frames, filenames) in enumerate(eval_loader):

      batch = Variable(batch.cuda(), volatile=True)

      original, out_imgs, losses, code_batch = eval_forward(
          model, (batch, ctx_frames), args)

      losses, msssim, psnr = finish_batch(
          args, filenames, original, out_imgs, 
          losses, code_batch, output_suffix)

      all_losses += losses
      all_msssim += msssim
      all_psnr += psnr

      if i % 10 == 0:
        print('\tevaluating iter %d (%f seconds)...' % (
          i, time.time() - start_time))

  return (np.array(all_losses).mean(axis=0),
          np.array(all_msssim).mean(axis=0),
          np.array(all_psnr).mean(axis=0))
