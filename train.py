import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable

from dataset import get_loader
from evaluate import run_eval
from train_options import parser
from util import get_models, init_lstm, set_train, set_eval
from util import prepare_inputs, forward_ctx

args = parser.parse_args()
print(args)

############### Data ###############
train_loader = get_loader(
  is_train=True,
  root=args.train, mv_dir=args.train_mv, 
  args=args
)

def get_eval_loaders():
  # We can extend this dict to evaluate on multiple datasets.
  eval_loaders = {
    'TVL': get_loader(
        is_train=False,
        root=args.eval, mv_dir=args.eval_mv,
        args=args),
  }
  return eval_loaders



############### Model ###############
encoder, binarizer, decoder, unet = get_models(
  args=args, v_compress=args.v_compress, 
  bits=args.bits,
  encoder_fuse_level=args.encoder_fuse_level,
  decoder_fuse_level=args.decoder_fuse_level)

nets = [encoder, binarizer, decoder]
if unet is not None:
  nets.append(unet)

gpus = [int(gpu) for gpu in args.gpus.split(',')]
if len(gpus) > 1:
  print("Using GPUs {}.".format(gpus))
  for net in nets:
    net = nn.DataParallel(net, device_ids=gpus)

params = [{'params': net.parameters()} for net in nets]

solver = optim.Adam(
    params,
    lr=args.lr)

milestones = [int(s) for s in args.schedule.split(',')]
scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=args.gamma)

if not os.path.exists(args.model_dir):
  print("Creating directory %s." % args.model_dir)
  os.makedirs(args.model_dir)

############### Checkpoints ###############
def resume(index):
  names = ['encoder', 'binarizer', 'decoder', 'unet']

  for net_idx, net in enumerate(nets):
    if net is not None:
      name = names[net_idx]
      checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
          args.model_dir, args.save_model_name, 
          name, index)

      print('Loading %s from %s...' % (name, checkpoint_path))
      net.load_state_dict(torch.load(checkpoint_path))


def save(index):
  names = ['encoder', 'binarizer', 'decoder', 'unet']

  for net_idx, net in enumerate(nets):
    if net is not None:
      torch.save(encoder.state_dict(), 
                 '{}/{}_{}_{:08d}.pth'.format(
                   args.model_dir, args.save_model_name, 
                   names[net_idx], index))


############### Training ###############

train_iter = 0
just_resumed = False
if args.load_model_name:
    print('Loading %s@iter %d' % (args.load_model_name,
                                  args.load_iter))

    resume(args.load_model_name, args.load_iter)
    train_iter = args.load_iter
    scheduler.last_epoch = train_iter - 1
    just_resumed = True


while True:

    for batch, (crops, ctx_frames, _) in enumerate(train_loader):
        scheduler.step()
        train_iter += 1

        if train_iter > args.max_train_iters:
          break

        batch_t0 = time.time()

        solver.zero_grad()

        # Init LSTM states.
        (encoder_h_1, encoder_h_2, encoder_h_3,
         decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm(
            batch_size=(crops[0].size(0) * args.num_crops), height=crops[0].size(2),
            width=crops[0].size(3), args=args)

        # Forward U-net.
        if args.v_compress:
            unet_output1, unet_output2 = forward_ctx(unet, ctx_frames)
        else:
            unet_output1 = Variable(torch.zeros(args.batch_size,)).cuda()
            unet_output2 = Variable(torch.zeros(args.batch_size,)).cuda()

        res, frame1, frame2, warped_unet_output1, warped_unet_output2 = prepare_inputs(
            crops, args, unet_output1, unet_output2)

        losses = []

        bp_t0 = time.time()
        _, _, height, width = res.size()

        out_img = torch.zeros(1, 3, height, width).cuda() + 0.5

        for _ in range(args.iterations):
            if args.v_compress and args.stack:
                encoder_input = torch.cat([frame1, res, frame2], dim=1)
            else:
                encoder_input = res

            # Encode.
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                encoder_input, encoder_h_1, encoder_h_2, encoder_h_3,
                warped_unet_output1, warped_unet_output2)

            # Binarize.
            codes = binarizer(encoded)

            # Decode.
            (output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = decoder(
                codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,
                warped_unet_output1, warped_unet_output2)

            res = res - output
            out_img = out_img + output.data
            losses.append(res.abs().mean())

        bp_t1 = time.time()

        loss = sum(losses) / args.iterations
        loss.backward()

        for net in [encoder, binarizer, decoder, unet]:
            if net is not None:
                torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)

        solver.step()

        batch_t1 = time.time()

        print(
            '[TRAIN] Iter[{}]; LR: {}; Loss: {:.6f}; Backprop: {:.4f} sec; Batch: {:.4f} sec'.
            format(train_iter, 
                   scheduler.get_lr()[0], 
                   loss.data[0],
                   bp_t1 - bp_t0, 
                   batch_t1 - batch_t0))

        if train_iter % 100 == 0:
            print('Loss at each step:')
            print(('{:.4f} ' * args.iterations +
                   '\n').format(* [l.data[0] for l in losses]))

        if train_iter % args.checkpoint_iters == 0:
            save(train_iter)

        if just_resumed or train_iter % args.eval_iters == 0 or train_iter == 100:
            print('Start evaluation...')

            set_eval(nets)

            eval_loaders = get_eval_loaders()
            for eval_name, eval_loader in eval_loaders.items():
                eval_begin = time.time()
                eval_loss, mssim, psnr = run_eval(nets, eval_loader, args,
                    output_suffix='iter%d' % train_iter)

                print('Evaluation @iter %d done in %d secs' % (
                    train_iter, time.time() - eval_begin))
                print('%s Loss   : ' % eval_name
                      + '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
                print('%s MS-SSIM: ' % eval_name
                      + '\t'.join(['%.5f' % el for el in mssim.tolist()]))
                print('%s PSNR   : ' % eval_name
                      + '\t'.join(['%.5f' % el for el in psnr.tolist()]))

            set_train(nets)
            just_resumed = False


    if train_iter > args.max_train_iters:
      print('Training done.')
      break
