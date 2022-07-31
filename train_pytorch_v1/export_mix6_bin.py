from dataset import trainset
from model import ModelDic
from model import boardH, boardW

import argparse
import glob
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import torch.nn as nn
import torch
import os
import time
import shutil
import lz4.frame
from zlib import crc32

try:
    os.mkdir("export")
except:
    pass
else:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='which gpu')
    parser.add_argument('--cpu', action='store_true', default=False, help='whether use cpu')
    parser.add_argument('--model', type=str, default='test', help='model path')
    parser.add_argument('--export', type=str, default='', help='export path')
    parser.add_argument('--copy',
                        action='store_true',
                        default=False,
                        help='copy a backup for this model, for selfplay training')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    if (args.cpu):
        device = torch.device('cpu')
    modelname = args.model
    exportname = args.export
    if (exportname == ''):
        exportname = modelname

    file_path = f'saved_models/{modelname}.pth'
    if os.path.exists(file_path):
        data = torch.load(file_path, map_location=device)
        model_type = data['model_type']
        model_param = data['model_param']
        model = ModelDic[model_type](*model_param).to(device)

        model.load_state_dict(data['state_dict'])
        totalstep = data['totalstep']
        print(f"loaded model: type={model_type}, param={model.model_param}, totalstep={totalstep}")
    else:
        print(f"Invalid Model Path: {file_path}")
        exit(0)

    #copy model file
    if (args.copy):
        modeldir = 'export/' + modelname
        try:
            os.mkdir(modeldir)
        except:
            pass
        else:
            pass
        modelDestPath = modeldir + '/' + str(totalstep) + '.pth'
        shutil.copy(file_path, modelDestPath)

    model.eval()

    time0 = time.time()
    _, pc, vc, _ = model_param
    print(f"Start: pc={pc}, vc={vc}")
    exportPath = 'export/' + exportname + '.bin.lz4'
    exportfile = lz4.frame.open(
        exportPath,
        'wb',
        compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC,
        content_checksum=True,
    )

    #-------------------------------------------------------------
    #write bin header

    MAGIC = crc32(b'gomoku network weight version 1')  # 0xacd8cc6a
    arch_hash = 0x78f3c05b  # for pc=16 and vc=32 mix6net
    rule_mask = (0x4) & 0xffffffff  # 0x1: freestyle, 0x2: standard, 0x4: renju
    boardsize_mask = (1 << (boardH - 1)) & 0xffffffff  # assume boardH==boardW
    description = f'model={model_type}; param={model_param}; totalstep={totalstep}'
    encoded_description = description.encode('utf-8')

    exportfile.write(MAGIC.to_bytes(4, byteorder='little', signed=False))
    exportfile.write(arch_hash.to_bytes(4, byteorder='little', signed=False))
    exportfile.write(rule_mask.to_bytes(4, byteorder='little', signed=False))
    exportfile.write(boardsize_mask.to_bytes(4, byteorder='little', signed=False))
    exportfile.write(len(encoded_description).to_bytes(4, byteorder='little', signed=False))
    exportfile.write(encoded_description)

    #export featuremap
    #-------------------------------------------------------------
    scale_now = 1
    print("Exporting FeatureMap")

    #prepare data
    def fullData(length):  # 三进制全部排列
        x = np.zeros((1, length), dtype=np.int64)
        for i in range(length):
            x1 = x.copy()
            x1[:, i] = 1
            x2 = x.copy()
            x2[:, i] = 2
            x = np.concatenate((x, x1, x2), axis=0)
        return x

    buf = np.ones((4 * (3**11), pc + vc), dtype=np.float64) * 114514

    pow3 = np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561, 3**9, 3**10, 3**11], dtype=np.int64)
    pow3 = pow3[:, np.newaxis]
    #无边界和单边边界 长度9

    data = fullData(11)
    label = model.exportMapTable(data, device)

    #无边界与正边界
    for r in range(6):
        ids = np.matmul(data[:, r:], pow3[:11 - r]) + pow3[12 - r:].sum()
        for i in range(ids.shape[0]):
            buf[ids[i]] = label[:, i, 5 + r]

    #负边界
    for l in range(1, 6):
        ids = np.matmul(data[:, :11 - l], pow3[l:-1]) + 2 * 3**11 + pow3[:l - 1].sum()
        for i in range(ids.shape[0]):
            buf[ids[i]] = label[:, i, 5 - l]

    #双边边界
    for left in range(1, 6):
        for right in range(1, 6):
            L = 11 - left - right
            data = fullData(L)
            label = model.exportMapTable(data, device)
            idbias = 3 * 3**11 + pow3[0:left - 1].sum((0, 1)) + pow3[12 - right:-1].sum((0, 1))
            ids = np.matmul(data, pow3[left:11 - right]) + idbias
            for i in range(ids.shape[0]):
                buf[ids[i]] = label[:, i, 5 - left]

    bound = 1  #这个变量存储上界，时刻注意int16溢出
    useful = (buf[:, 0] != 114514)
    buf = buf * (useful[:, np.newaxis].astype(np.float64))
    usefulcount = useful.sum()
    print("Useful=", usefulcount)
    wmax = np.abs(buf).max()
    bound *= wmax
    print("Max=", wmax)

    #np.set_printoptions(suppress=True,precision=3)
    w_scale = 6000 / wmax
    scale_now *= w_scale  #200
    maxint = wmax * w_scale
    if (maxint > 32700):
        print("Error! Maxint=", maxint)
        exit(0)
    buf[~useful] = 0  # clear not used buf
    print("feature_map", buf.shape)
    assert np.all(np.abs(buf * w_scale) < 32767)
    exportfile.write((buf * w_scale).astype('<i2').tobytes())  # (708588, 48)

    bound *= w_scale
    print("Bound=", bound)

    # export others
    print("Finished featuremap, now exploring others")
    # -------------------------------------------------------------
    #1 map_leakyrelu
    #prepare
    scale_maplr = scale_now  #prepare for value head

    map_lr_slope = model.map_leakyrelu.slope.data.cpu().numpy()
    map_lr_slope = np.tanh(map_lr_slope / 6) * 6
    map_lr_bias = model.map_leakyrelu.bias.data.cpu().numpy()

    #calculate max
    wmax = np.abs(map_lr_slope).max()
    bmax = np.abs(map_lr_bias).max()
    bound_c = bound * (np.abs(map_lr_slope) + 1) + np.abs(map_lr_bias) * scale_now
    print("map lr maxslope=", wmax)
    print("map lr maxbias=", bmax)
    map_lr_slope_sub1 = (map_lr_slope - 1) * 0.125  #slope>1会溢出，所以负半轴乘slope*0.125再乘2，正半轴乘0.25
    map_lr_bias = map_lr_bias * scale_now

    maxint = max((wmax + 1) * 2**15 / 8, bmax * scale_now)  #mulhrs右移15位
    if (maxint > 32700):
        print("Error! Maxint=", maxint)
        exit(0)

    #write
    print("map_lr_slope_sub1", map_lr_slope_sub1.shape)
    print("map_lr_bias", map_lr_bias.shape)
    assert np.all(np.abs(map_lr_slope_sub1 * 2**15) < 32767)
    assert np.all(np.abs(map_lr_bias) < 32767)
    exportfile.write((map_lr_slope_sub1 * 2**15).astype('<i2').tobytes())  # (48,)
    exportfile.write(map_lr_bias.astype('<i2').tobytes())  # (48,)

    #bound

    bound = bound * (wmax + 1) + bmax * scale_now
    print("Bound=", bound)

    # 2 policyConv

    policyConvWeight = model.policy_conv.weight.data.cpu().numpy()
    policyConvBias = model.policy_conv.bias.data.cpu().numpy()

    #calculate max
    wmax = np.abs(policyConvWeight).max()
    bound_c = (np.abs(policyConvWeight).sum(
        (1, 2, 3)) * bound_c[:pc] + np.abs(policyConvBias) * scale_now)
    bmax = np.abs(policyConvBias).max()
    print("policyConvWeight max=", wmax)
    print("policyConvBias max=", bmax)

    maxint = max(wmax * 2**15, bmax * scale_now)

    w_scale = min(32700 / maxint, 32700 / bound_c.max())
    print("policy conv w_scale=", w_scale)
    scale_now *= w_scale
    bound_c *= w_scale

    policyConvWeight = policyConvWeight * w_scale
    policyConvBias = policyConvBias * scale_now

    maxint = max(wmax * w_scale * 2**15, bmax * scale_now)
    if (maxint > 32750):
        print("Error! Maxint=", maxint)
        exit(0)

    #write
    print("policyConvWeight", policyConvWeight.shape)
    print("policyConvBias", policyConvBias.shape)
    assert np.all(np.abs(policyConvWeight * 2**15) < 32767)
    assert np.all(np.abs(policyConvBias) < 32767)
    exportfile.write((policyConvWeight.reshape(pc, 3, 3).transpose(
        (1, 2, 0)) * 2**15).astype('<i2').tobytes())  # (3, 3, 16)
    exportfile.write(policyConvBias.astype('<i2').tobytes())  # (16,)

    #bound
    print("Boundc=", bound_c.max())
    print("if boundc < 32767, that's ok")

    # 2 policyFinalConv
    #prepare
    policyFinalConv = model.policy_linear.weight.data.cpu().numpy()

    #calculate max
    wmax = np.abs(policyFinalConv).max()
    print("policyFinalConv max=", wmax)

    maxint = wmax * 2**15
    w_scale = min(32700 / maxint, 0.5)
    scale_now *= w_scale
    if (maxint * w_scale > 32750):
        print("Error! Maxint=", maxint)
        exit(0)

    #write
    policyFinalConv = policyFinalConv * w_scale
    print("policyFinalConv", policyFinalConv.shape)
    assert np.all(np.abs(policyFinalConv * 2**15) < 32767)
    exportfile.write((policyFinalConv.reshape(pc, ) * 2**15).astype('<i2').tobytes())  # (16,)

    #bound
    bound = (np.abs(policyFinalConv)[0, :, 0, 0] * bound_c).sum()
    print("Bound=", bound)
    print("If this bound is a little bigger than 32767, there's no big problem")

    #剩下的都是float 无需担心量化

    #policy final leakyrelu
    p_slope = model.policy_leakyrelu.slope.data.cpu().numpy()[0]
    policy_neg_slope = p_slope / scale_now
    policy_pos_slope = 1 / scale_now
    print("policy_neg_slope", policy_neg_slope)
    print("policy_pos_slope", policy_pos_slope)
    exportfile.write(np.array([policy_neg_slope, policy_pos_slope], dtype='<f4').tobytes())

    #scale_beforemlp
    scale_beforemlp = 1 / scale_maplr # / boardH / boardW
    print("scale_beforemlp", scale_beforemlp)
    exportfile.write(np.array([scale_beforemlp], dtype='<f4').tobytes())

    # 从这里开始 scale就是1了
    #value first leakyrelu
    value_lr_slope = model.value_leakyrelu.slope.data.cpu().numpy()
    value_lr_slope_sub1 = value_lr_slope - 1

    print("value_lr_slope_sub1", value_lr_slope_sub1.shape, value_lr_slope_sub1.min(),
          value_lr_slope_sub1.mean(), value_lr_slope_sub1.max())
    exportfile.write(value_lr_slope_sub1.astype('<f4').tobytes())  # (32,)

    #mlp layer 1
    #scale_layer1=1/scale_maplr/boardH/boardW
    w = model.value_linear1.weight.data.cpu().numpy()
    #w=w*scale_layer1
    b = model.value_linear1.bias.data.cpu().numpy()

    print("mlp_w1", w.shape, w.min(), w.mean(), w.max())
    print("mlp_b1", b.shape, b.min(), b.mean(), b.max())
    exportfile.write(w.T.astype('<f4').tobytes())  # (32, 32)
    exportfile.write(b.astype('<f4').tobytes())  # (32,)

    # mlp layer 2
    w = model.value_linear2.weight.data.cpu().numpy()
    b = model.value_linear2.bias.data.cpu().numpy()

    print("mlp_w2", w.shape, w.min(), w.mean(), w.max())
    print("mlp_b2", b.shape, b.min(), b.mean(), b.max())
    exportfile.write(w.T.astype('<f4').tobytes())  # (32, 32)
    exportfile.write(b.astype('<f4').tobytes())  # (32,)

    # mlp layer 3
    w = model.value_linearfinal.weight.data.cpu().numpy()
    b = model.value_linearfinal.bias.data.cpu().numpy()

    print("mlp_w3", w.shape, w.min(), w.mean(), w.max())
    print("mlp_b3", b.shape, b.min(), b.mean(), b.max())

    exportfile.write(w.T.astype('<f4').tobytes())  # (32, 3)
    exportfile.write(np.zeros(5, dtype='<f4').tobytes())  # padding: (5,)
    exportfile.write(b.astype('<f4').tobytes())  # (3,)
    exportfile.write(np.zeros(5, dtype='<f4').tobytes())  # padding: (5,)

    exportfile.close()

    #copy txt file
    if (args.copy):
        #modeldir='export/'+modelname
        exportCopyDestPath = modeldir + '/' + str(totalstep) + '.bin.lz4'
        shutil.copy(exportPath, exportCopyDestPath)

    print("success")
