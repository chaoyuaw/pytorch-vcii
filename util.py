from collections import namedtuple
from scipy.misc import imsave
import cv2
import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import network
from metric import msssim, psnr
from unet import UNet


def get_models(args, v_compress, bits, encoder_fuse_level, decoder_fuse_level):

    encoder = network.EncoderCell(
        v_compress=v_compress,
        stack=args.stack,
        fuse_encoder=args.fuse_encoder,
        fuse_level=encoder_fuse_level
    ).cuda()

    binarizer = network.Binarizer(bits).cuda()

    decoder = network.DecoderCell(
        v_compress=v_compress, shrink=args.shrink,
        bits=bits,
        fuse_level=decoder_fuse_level
    ).cuda()

    if v_compress:
        unet = UNet(3, args.shrink).cuda()
    else:
        unet = None

    return encoder, binarizer, decoder, unet


def get_identity_grid(size):
    id_mat = Variable(torch.FloatTensor([[1, 0, 0, 0, 1, 0]] * size[0]), 
        requires_grad=False).view(-1, 2, 3).cuda()
    return F.affine_grid(id_mat, size)


def transpose_to_grid(frame2):
    # b, c, h, w
    # b, h, c, w
    # b, h, w, c
    frame2 = frame2.transpose(1, 2)
    frame2 = frame2.transpose(2, 3)
    return frame2


def get_id_grids(size):
    batch_size, _, height, width = size
    # The 32 here is not used.
    id_grid_4 = get_identity_grid(
        torch.Size([batch_size, 32, height//2, width//2]))
    id_grid_3 = get_identity_grid(
        torch.Size([batch_size, 32, height//4, width//4]))
    id_grid_2 = get_identity_grid(
        torch.Size([batch_size, 32, height//8, width//8]))
    return id_grid_4, id_grid_3, id_grid_2


def get_large_id_grid(size):
    batch_size, _, height, width = size
    # The 32 here is not used.
    return get_identity_grid(
        torch.Size([batch_size, 32, height, width]))


down_sample = nn.AvgPool2d(2, stride=2)


def get_flows(flow):
    flow_4 = down_sample(flow)
    flow_3 = down_sample(flow_4)
    flow_2 = down_sample(flow_3)

    flow_4 = transpose_to_grid(flow_4)
    flow_3 = transpose_to_grid(flow_3)
    flow_2 = transpose_to_grid(flow_2)

    final_grid_4 = flow_4 + 0.5
    final_grid_3 = flow_3 + 0.5
    final_grid_2 = flow_2 + 0.5

    return [final_grid_4, final_grid_3, final_grid_2]


def prepare_batch(batch, v_compress, warp):
    res = batch - 0.5

    flows = []
    frame1, frame2 = None, None
    if v_compress:
        if warp:
            assert res.size(1) == 13
            flow_1 = res[:, 9:11]
            flow_2 = res[:, 11:13]

            flows.append(get_flows(flow_1))
            flows.append(get_flows(flow_2))

        frame1 = res[:, :3]
        frame2 = res[:, 6:9]
        res = res[:, 3:6]
    return res, frame1, frame2, flows


def set_eval(models):
    for m in models:
        if m is not None:
            m.eval()


def set_train(models):
    for m in models:
        if m is not None:
            m.train()


def eval_forward(model, batch, args):
    batch, ctx_frames = batch
    cooked_batch = prepare_batch(
        batch, args.v_compress, args.warp)


    return forward_model(
        model=model,
        cooked_batch=cooked_batch,
        ctx_frames=ctx_frames,
        args=args,
        v_compress=args.v_compress,
        iterations=args.iterations,
        encoder_fuse_level=args.encoder_fuse_level,
        decoder_fuse_level=args.decoder_fuse_level)


def prepare_unet_output(unet, unet_input, flows, warp):
    unet_output1, unet_output2 = [], []
    unet_outputs = unet(unet_input)
    for u_out in unet_outputs:
        u_out1, u_out2 = u_out.chunk(2, dim=0)
        unet_output1.append(u_out1)
        unet_output2.append(u_out2)
    if warp:
        unet_output1, unet_output2 = warp_unet_outputs(
            flows, unet_output1, unet_output2)
    return unet_output1, unet_output2


def prepare_inputs(crops, args, unet_output1, unet_output2):
    data_arr = []
    frame1_arr = []
    frame2_arr = []
    warped_unet_output1 = []
    warped_unet_output2 = []

    for crop_idx, data in enumerate(crops):
        patches = Variable(data.cuda())

        res, frame1, frame2, flows = prepare_batch(patches, args.v_compress, args.warp)
        data_arr.append(res)
        frame1_arr.append(frame1)
        frame2_arr.append(frame2)

        if args.v_compress and args.warp:
            wuo1, wuo2 = warp_unet_outputs(
                flows, unet_output1, unet_output2)

            warped_unet_output1.append(wuo1)
            warped_unet_output2.append(wuo2)


    res = torch.cat(data_arr, dim=0)
    frame1 = torch.cat(frame1_arr, dim=0)
    frame2 = torch.cat(frame2_arr, dim=0)
    warped_unet_output1 = [torch.cat(wuos, dim=0) for wuos in zip(*warped_unet_output1)]
    warped_unet_output2 = [torch.cat(wuos, dim=0) for wuos in zip(*warped_unet_output2)]

    return res, frame1, frame2, warped_unet_output1, warped_unet_output2


def forward_ctx(unet, ctx_frames):
    ctx_frames = Variable(ctx_frames.cuda()) - 0.5
    frame1 = ctx_frames[:, :3]
    frame2 = ctx_frames[:, 3:]

    unet_output1, unet_output2 = [], []

    unet_outputs = unet(torch.cat([frame1, frame2], dim=0))
    for u_out in unet_outputs:
        u_out1, u_out2 = u_out.chunk(2, dim=0)
        unet_output1.append(u_out1)
        unet_output2.append(u_out2)

    return unet_output1, unet_output2


def forward_model(model, cooked_batch, ctx_frames, args, v_compress,
                  iterations, encoder_fuse_level, decoder_fuse_level):
    encoder, binarizer, decoder, unet = model
    res, _, _, flows = cooked_batch

    ctx_frames = Variable(ctx_frames.cuda()) - 0.5
    frame1 = ctx_frames[:, :3]
    frame2 = ctx_frames[:, 3:]

    init_rnn = init_lstm

    batch_size, _, height, width = res.size()
    (encoder_h_1, encoder_h_2, encoder_h_3,
     decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_rnn(batch_size,
                                                                      height,
                                                                      width,
                                                                      args)

    original = res.data.cpu().numpy() + 0.5

    out_img = torch.zeros(1, 3, height, width) + 0.5
    out_imgs = []
    losses = []

    # UNet.
    enc_unet_output1 = Variable(torch.zeros(args.batch_size,), volatile=True).cuda()
    enc_unet_output2 = Variable(torch.zeros(args.batch_size,), volatile=True).cuda()

    dec_unet_output1 = Variable(torch.zeros(args.batch_size,), volatile=True).cuda()
    dec_unet_output2 = Variable(torch.zeros(args.batch_size,), volatile=True).cuda()
    if v_compress:
        # Use decoded context frames to decode.
        dec_unet_output1, dec_unet_output2 =  prepare_unet_output(
            unet, torch.cat([frame1, frame2], dim=0), flows, warp=args.warp)

        enc_unet_output1, enc_unet_output2 = dec_unet_output1, dec_unet_output2

        assert len(enc_unet_output1) == 3 and len(enc_unet_output2) == 3, (len(enc_unet_output1), len(enc_unet_output2))
        assert len(dec_unet_output1) == 3 and len(dec_unet_output2) == 3, (len(dec_unet_output1), len(dec_unet_output2))
        for jj in range(3 - max(encoder_fuse_level, decoder_fuse_level)):
            enc_unet_output1[jj] = None
            enc_unet_output2[jj] = None
            dec_unet_output1[jj] = None
            dec_unet_output2[jj] = None

    codes = []
    prev_psnr = 0.0
    for _ in range(iterations):

        if args.v_compress and args.stack:
            encoder_input = torch.cat([frame1, res, frame2], dim=1)
        else:
            encoder_input = res

        # Encode.
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            encoder_input, encoder_h_1, encoder_h_2, encoder_h_3,
            enc_unet_output1, enc_unet_output2)

        # Binarize.
        code = binarizer(encoded)
        if args.save_codes:
            codes.append(code.data.cpu().numpy())

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,
            dec_unet_output1, dec_unet_output2)

        res = res - output
        out_img = out_img + output.data.cpu()
        out_img_np = out_img.numpy().clip(0, 1)

        out_imgs.append(out_img_np)
        losses.append(float(res.abs().mean().data.cpu().numpy()))

    return original, np.array(out_imgs), np.array(losses), np.array(codes)


def save_numpy_array_as_image(filename, arr):
    imsave(
        filename, 
        np.squeeze(arr * 255.0).astype(np.uint8)
        .transpose(1, 2, 0))


def save_torch_array_as_image(filename, arr):
    imsave(
        filename, 
        np.squeeze(arr.numpy().clip(0, 1) * 255.0).astype(np.uint8)
        .transpose(1, 2, 0))


def evaluate(original, out_imgs):

    ms_ssims = np.array([get_ms_ssim(original, out_img) for out_img in out_imgs])
    psnrs    = np.array([   get_psnr(original, out_img) for out_img in out_imgs])

    return ms_ssims, psnrs


def evaluate_all(original, out_imgs):

    all_msssim, all_psnr = [], []
    for j in range(original.shape[0]):
        msssim, psnr = evaluate(
            original[None, j],
            [out_img[None, j] for out_img in out_imgs])
        all_msssim.append(msssim)
        all_psnr.append(psnr)

    return all_msssim, all_psnr


def as_img_array(image):
    # Iutput: [batch_size, depth, height, width]
    # Output: [batch_size, height, width, depth]
    image = image.clip(0, 1) * 255.0
    return image.astype(np.uint8).transpose(0, 2, 3, 1)


def get_ms_ssim(original, compared):
    return msssim(as_img_array(original), as_img_array(compared))


def get_psnr(original, compared):
    return psnr(as_img_array(original), as_img_array(compared))


def warp_unet_outputs(flows, unet_output1, unet_output2):
    [grid_1_4, grid_1_3, grid_1_2] = flows[0]
    [grid_2_4, grid_2_3, grid_2_2] = flows[1]

    warped_unet_output1, warped_unet_output2 = [], []

    warped_unet_output1.append(F.grid_sample(
        unet_output1[0], grid_1_2, padding_mode='border'))
    warped_unet_output2.append(F.grid_sample(
        unet_output2[0], grid_2_2, padding_mode='border'))

    warped_unet_output1.append(F.grid_sample(
        unet_output1[1], grid_1_3, padding_mode='border'))
    warped_unet_output2.append(F.grid_sample(
        unet_output2[1], grid_2_3, padding_mode='border'))

    warped_unet_output1.append(F.grid_sample(
        unet_output1[2], grid_1_4, padding_mode='border'))
    warped_unet_output2.append(F.grid_sample(
        unet_output2[2], grid_2_4, padding_mode='border'))

    return warped_unet_output1, warped_unet_output2


def init_lstm(batch_size, height, width, args):

    encoder_h_1 = (Variable(
        torch.zeros(batch_size, 256, height // 4, width // 4)),
                   Variable(
                       torch.zeros(batch_size, 256, height // 4, width // 4)))
    encoder_h_2 = (Variable(
        torch.zeros(batch_size, 512, height // 8, width // 8)),
                   Variable(
                       torch.zeros(batch_size, 512, height // 8, width // 8)))
    encoder_h_3 = (Variable(
        torch.zeros(batch_size, 512, height // 16, width // 16)),
                   Variable(
                       torch.zeros(batch_size, 512, height // 16, width // 16)))

    decoder_h_1 = (Variable(
        torch.zeros(batch_size, 512, height // 16, width // 16)),
                   Variable(
                       torch.zeros(batch_size, 512, height // 16, width // 16)))
    decoder_h_2 = (Variable(
        torch.zeros(batch_size, 512, height // 8, width // 8)),
                   Variable(
                       torch.zeros(batch_size, 512, height // 8, width // 8)))
    decoder_h_3 = (Variable(
        torch.zeros(batch_size, 256, height // 4, width // 4)),
                   Variable(
                       torch.zeros(batch_size, 256, height // 4, width // 4)))
    decoder_h_4 = (Variable(
        torch.zeros(batch_size, 256 if False else 128, height // 2, width // 2)),
                   Variable(
                       torch.zeros(batch_size, 256 if False else 128, height // 2, width // 2)))

    encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
    encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
    encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

    return (encoder_h_1, encoder_h_2, encoder_h_3, 
            decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)


