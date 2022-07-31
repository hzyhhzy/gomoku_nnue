from dataset import trainset
from model import Model_v2
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
    os.mkdir("../export")
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

    file_path = f'../saved_models/{modelname}/model.pth'
    model_type = None
    if os.path.exists(file_path):
        data = torch.load(file_path, map_location=device)
        model_type = data['model_type']
        model_param = data['model_param']
        if (model_type != "v2" and model_type != "v2beta0"):
            print(f"Invalid Model Type: {model_type}")
            exit(0)
        model = Model_v2(*model_param).to(device)

        model.load_state_dict(data['state_dict'])
        totalstep = data['totalstep']
        print(f"loaded model: type={model_type}, param={model.model_param}, totalstep={totalstep}")
    else:
        print(f"Invalid Model Path: {file_path}")
        exit(0)

    #copy model file
    if (args.copy):
        modeldir = '../export/' + modelname
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
    _, _, groupc, mlpc, _ = model_param
    print(f"Start: groupc={groupc}, mlpc={mlpc}")
    exportPath = '../export/' + exportname + '.bin.lz4'
    exportfile = lz4.frame.open(
        exportPath,
        'wb',
        compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC,
        content_checksum=True,
    )

    #-------------------------------------------------------------
    # write bin header

    MAGIC = crc32(b'gomoku network weight version 1')  # 0xacd8cc6a
    arch_hash = crc32(b'nnuev2') ^ (groupc | (mlpc << 16))
    rule_mask = (0x1) & 0xffffffff  # 0x1: freestyle, 0x2: standard, 0x4: renju
    boardsize_mask = (1 << (boardH - 1)) & 0xffffffff  # assume boardH==boardW
    description = f'model={model_type}; param={model_param}; totalstep={totalstep}'
    encoded_description = description.encode('utf-8')

    exportfile.write(MAGIC.to_bytes(4, byteorder='little', signed=False))
    exportfile.write(arch_hash.to_bytes(4, byteorder='little', signed=False))
    exportfile.write(rule_mask.to_bytes(4, byteorder='little', signed=False))
    exportfile.write(boardsize_mask.to_bytes(4, byteorder='little', signed=False))
    exportfile.write(len(encoded_description).to_bytes(4, byteorder='little', signed=False))
    exportfile.write(encoded_description)

    # export featuremap
    #-------------------------------------------------------------
    print("Exporting mapping")

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

    buf = np.ones((4 * (3**11), groupc * 2), dtype=np.float64) * 114514

    pow3 = np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561, 3**9, 3**10, 3**11], dtype=np.int64)
    pow3 = pow3[:, np.newaxis]
    #无边界和单边边界 长度11

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

    useful = (buf[:, 0] != 114514)
    buf = buf * (useful[:, np.newaxis].astype(np.float64))
    usefulcount = useful.sum()
    print("Useful=", usefulcount)

    scale_now = 1  #这个变量存储int16和float的换算比
    bound = np.abs(buf).max()  #这个变量存储上界，时刻注意int16溢出
    print("Max=", bound)

    map_maxint = 3000
    g2extrascale = 0.5
    trunkconv1_extrabound = 3  #允许少量超越bound
    trunkconv2_extrabound = 9

    w_scale = map_maxint / bound  #w_scale表示这一步的倍数

    scale_now *= w_scale
    bound *= w_scale
    g2scale = scale_now * g2extrascale
    g2bound = bound * g2scale / scale_now
    maxint = bound  #maxint表示导出权重的最大值,此处恰好为bound
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)
    buf[~useful] = 0  # clear not used buf
    print("feature_map", buf.shape)
    buf_quant = buf.copy()
    buf_quant[:, :groupc] *= w_scale
    buf_quant[:, groupc:] *= g2scale
    assert np.all(np.abs(buf_quant) < 32767)
    exportfile.write(buf_quant.astype('<i2').tobytes())  # (708588, 128)

    print("Bound after mapping = ", bound)
    print("Scale after mapping = ", scale_now)

    # export others
    print("Finished mapping, now exporting others")
    # -------------------------------------------------------------
    #1 g1lr
    #pytorch里面是4线平均，c++里是4线求和
    #g1lr本身对scale和bound无影响，w_scale=1
    scale_now *= 4
    bound *= 4

    g1lr_w = model.g1lr.export_slope()

    maxint = np.abs(g1lr_w * 2**15).max()  #mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    #write
    print("g1lr_w", g1lr_w.shape)
    assert np.all(np.abs(g1lr_w * 2**15) < 32767)
    exportfile.write((g1lr_w * 2**15).astype('<i2').tobytes())  # (64,)

    print("Bound after g1lr = ", bound)
    print("Scale after g1lr = ", scale_now)

    #2 h1conv

    h1conv_w = model.h1conv.w.data.cpu().numpy()
    h1conv_w = np.squeeze(h1conv_w, 2)
    h1conv_w = (h1conv_w + np.flip(h1conv_w, 0)) / 2  #symmetry
    assert (h1conv_w.shape == (11, groupc))
    h1conv_b = model.h1conv.b.data.cpu().numpy()

    h1conv_b *= scale_now
    bound = (np.abs(h1conv_w).sum(0) * bound + np.abs(h1conv_b)).max()

    w_scale = g2scale / scale_now
    scale_now *= w_scale
    bound *= w_scale
    h1conv_w *= w_scale
    h1conv_b *= w_scale

    maxint = max(np.abs(h1conv_w).max() * 2**15, np.abs(h1conv_b).max())
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    #write
    h1conv_w_quant = h1conv_w[5:5 + 6, :] * 2**15
    print("h1conv_w", h1conv_w_quant.shape)
    assert np.all(np.abs(h1conv_w_quant) < 32767)
    exportfile.write(h1conv_w_quant.astype('<i2').tobytes())

    print("h1conv_b", h1conv_b.shape)
    assert np.all(np.abs(h1conv_b) < 32767)
    exportfile.write(h1conv_b.astype('<i2').tobytes())

    print("Bound after h1conv = ", bound)
    print("Scale after h1conv = ", scale_now)

    #3 h1lr1
    # h1lr1本身对scale和bound无影响，w_scale=1
    h1lr1_w = model.h1lr1.export_slope()

    maxint = np.abs(h1lr1_w * 2**15).max()  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("h1lr1_w", h1lr1_w.shape)
    assert np.all(np.abs(h1lr1_w * 2**15) < 32767)
    exportfile.write((h1lr1_w * 2**15).astype('<i2').tobytes())  # mulhrs右移15位

    print("Bound after h1lr1 = ", bound)
    print("Scale after h1lr1 = ", scale_now)

    #+g2
    assert (scale_now == g2scale)
    bound += g2bound

    #4 h1lr2
    # h1lr2本身对scale和bound无影响，w_scale=1
    h1lr2_w = model.h1lr2.export_slope()

    maxint = np.abs(h1lr2_w * 2**15).max()  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("h1lr2_w", h1lr2_w.shape)
    assert np.all(np.abs(h1lr2_w * 2**15) < 32767)
    exportfile.write((h1lr2_w * 2**15).astype('<i2').tobytes())  # mulhrs右移15位

    print("Bound after h1lr2 = ", bound)
    print("Scale after h1lr2 = ", scale_now)

    # 对于h3，pytorch里面是4线平均，c++里也是4线平均，所以保持不变
    scale_now *= 1
    bound *= 1

    #5 h3lr
    # 本身对scale无影响，w_scale=1
    h3lr_w = model.h3lr.export_slope()
    h3lr_b = model.h3lr.bias.data.cpu().numpy()
    h3lr_b *= scale_now

    bmax = np.abs(h3lr_b).max()
    bound += bmax
    maxint = max(np.abs(h3lr_w * 2**15).max(), bmax)  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("h3lr_w", h3lr_w.shape)
    assert np.all(np.abs(h3lr_w * 2**15) < 32767)
    exportfile.write((h3lr_w * 2**15).astype('<i2').tobytes())  # mulhrs右移15位

    print("h3lr_b", h3lr_b.shape)
    assert np.all(np.abs(h3lr_b) < 32767)
    exportfile.write((h3lr_b).astype('<i2').tobytes())

    print("Bound after h3lr = ", bound)
    print("Scale after h3lr = ", scale_now)

    # 对于h3，pytorch里面是4线平均，c++里也是4线平均，所以保持不变
    scale_now *= 1
    bound *= 1

    #6 trunkconv1

    trunkconv1_w = model.trunkconv1.weight.data.cpu().numpy()
    trunkconv1_w = np.squeeze(trunkconv1_w, (2, 3))
    assert (trunkconv1_w.shape == (groupc, 4))
    trunkconv1_b = model.trunkconv1.bias.data.cpu().numpy()

    trunkconv1_b *= scale_now
    bound = (np.abs(trunkconv1_w).sum(1) * bound + np.abs(trunkconv1_b)).max()

    maxint = max(np.abs(trunkconv1_w).max() * 2**15, np.abs(trunkconv1_b).max())
    w_scale = min(32700 * trunkconv1_extrabound / bound, 32700 / maxint)
    print("trunkconv1_w_scale = ", w_scale)
    scale_now *= w_scale
    bound *= w_scale
    trunkconv1_w *= w_scale
    trunkconv1_b *= w_scale

    maxint = max(np.abs(trunkconv1_w).max() * 2**15, np.abs(trunkconv1_b).max())
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # c++代码和pytorch代码的分组顺序不一样
    trunkconv1_w = trunkconv1_w.reshape((-1, 4, 4, 4))
    trunkconv1_w = trunkconv1_w.transpose(3, 0, 2, 1)
    trunkconv1_w = trunkconv1_w.reshape((4, -1))
    trunkconv1_b = trunkconv1_b.reshape((-1, 4, 4))
    trunkconv1_b = trunkconv1_b.transpose(0, 2, 1)
    trunkconv1_b = trunkconv1_b.reshape((-1, ))

    # write
    print("trunkconv1_w", trunkconv1_w.shape)
    assert np.all(np.abs(trunkconv1_w * 2**15) < 32767)
    exportfile.write((trunkconv1_w * 2**15).astype('<i2').tobytes())  # mulhrs右移15位

    print("trunkconv1_b", trunkconv1_b.shape)
    assert np.all(np.abs(trunkconv1_b) < 32767)
    exportfile.write((trunkconv1_b).astype('<i2').tobytes())

    print("Bound after trunkconv1 = ", bound)
    print("Scale after trunkconv1 = ", scale_now)

    #7 trunklr1
    # 本身对scale无影响，w_scale=1
    trunklr1_w = model.trunklr1.export_slope()

    maxint = np.abs(trunklr1_w * 2**15).max()  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("trunklr1_w", trunklr1_w.shape)
    assert np.all(np.abs(trunklr1_w * 2**15) < 32767)
    exportfile.write((trunklr1_w * 2**15).astype('<i2').tobytes())  # mulhrs右移15位

    print("Bound after trunklr1 = ", bound)
    print("Scale after trunklr1 = ", scale_now)
    print("实验表明大于100的数字的出现频率小于1/100000，因此scale小于300即可")

    #8 trunkconv2 trunklr2p trunklr2v

    trunkconv2_w = model.trunkconv2.w.data.cpu().numpy()
    trunkconv2_w = np.squeeze(trunkconv2_w, 2)
    assert (trunkconv2_w.shape == (3, groupc))

    trunklr2p_b = model.trunklr2p.bias.data.cpu().numpy()
    trunklr2v_b = model.trunklr2v.bias.data.cpu().numpy()
    trunklr2p_b *= scale_now
    trunklr2v_b *= scale_now

    trunklr2p_w = model.trunklr2p.export_slope()
    trunklr2v_w = model.trunklr2v.export_slope()

    bound = (np.maximum(np.abs(trunklr2p_b), np.abs(trunklr2v_b)) + bound *
             (np.abs(trunkconv2_w[0]) + 4 * np.abs(trunkconv2_w[1]) + 4 * np.abs(trunkconv2_w[2]))
             ).max()

    maxint = np.abs(trunkconv2_w).max() * 2**15
    w_scale = min(32700 * trunkconv2_extrabound / bound, 32700 / maxint)
    print("trunkconv2_w_scale = ", w_scale)
    scale_now *= w_scale
    bound *= w_scale
    trunkconv2_w *= w_scale
    trunklr2p_b *= w_scale
    trunklr2v_b *= w_scale

    maxint = np.abs(trunkconv2_w).max() * 2**15
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("trunkconv2_w", trunkconv2_w.shape)
    assert np.all(np.abs(trunkconv2_w * 2**15) < 32767)
    exportfile.write((trunkconv2_w * 2**15).astype('<i2').tobytes())

    print("trunklr2p_w", trunklr2p_w.shape)
    assert np.all(np.abs(trunklr2p_w * 2**15) < 32767)
    exportfile.write((trunklr2p_w * 2**15).astype('<i2').tobytes())

    print("trunklr2p_b", trunklr2p_b.shape)
    assert np.all(np.abs(trunklr2p_b) < 32767)
    exportfile.write(trunklr2p_b.astype('<i2').tobytes())

    print("trunklr2v_w", trunklr2v_w.shape)
    assert np.all(np.abs(trunklr2v_w * 2**15) < 32767)
    exportfile.write((trunklr2v_w * 2**15).astype('<i2').tobytes())

    print("trunklr2v_b", trunklr2v_b.shape)
    assert np.all(np.abs(trunklr2v_b) < 32767)
    exportfile.write(trunklr2v_b.astype('<i2').tobytes())

    print("Bound after trunkconv2+trunklr2 = ", bound)
    print("Scale after trunkconv2+trunklr2 = ", scale_now)
    print("实验表明大于100的数字的出现频率小于1/100000，因此scale小于300即可")

    #9 policy_linear

    policy_linear_w = model.policy_linear.weight.data.cpu().numpy()
    policy_linear_w = policy_linear_w.reshape(-1)

    maxint = np.abs(policy_linear_w).max() * 2**15
    w_scale = 32700 / maxint
    print("policy_linear_w_scale = ", w_scale)
    scale_policy = scale_now * w_scale
    policy_linear_w *= w_scale

    maxint = np.abs(policy_linear_w).max() * 2**15
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("policy_linear_w", policy_linear_w.shape)
    assert np.all(np.abs(policy_linear_w * 2**15) < 32767)
    exportfile.write((policy_linear_w * 2**15).astype('<i2').tobytes())

    print("scale_policyInv", 1 / scale_policy)
    exportfile.write(np.array([1 / scale_policy]).astype('<f4').tobytes())

    print("Scale after policy_linear_w = ", scale_policy)

    #10 value scale
    print("scale_beforemlpInv", 1 / scale_now)
    exportfile.write(np.array([1 / scale_now]).astype('<f4').tobytes())

    #从这里往下的都是float 无需担心量化，且scale=1

    # valuelr
    valuelr_w = model.valuelr.export_slope()
    valuelr_b = model.valuelr.bias.data.cpu().numpy()

    print("valuelr_w", valuelr_w.shape)
    exportfile.write(valuelr_w.astype('<f4').tobytes())
    print("valuelr_b", valuelr_b.shape)
    exportfile.write(valuelr_b.astype('<f4').tobytes())

    #mlp layer 1
    w = model.value_linear1.weight.data.cpu().numpy()
    b = model.value_linear1.bias.data.cpu().numpy()

    print("mlp_w1", w.shape, w.min(), w.mean(), w.max())
    print("mlp_b1", b.shape, b.min(), b.mean(), b.max())
    exportfile.write(w.T.astype('<f4').tobytes())
    exportfile.write(b.astype('<f4').tobytes())

    #mlp layer 2
    w = model.value_linear2.weight.data.cpu().numpy()
    b = model.value_linear2.bias.data.cpu().numpy()

    print("mlp_w2", w.shape, w.min(), w.mean(), w.max())
    print("mlp_b2", b.shape, b.min(), b.mean(), b.max())
    exportfile.write(w.T.astype('<f4').tobytes())
    exportfile.write(b.astype('<f4').tobytes())

    #mlp layer 3
    w = model.value_linear3.weight.data.cpu().numpy()
    b = model.value_linear3.bias.data.cpu().numpy()

    print("mlp_w3", w.shape, w.min(), w.mean(), w.max())
    print("mlp_b3", b.shape, b.min(), b.mean(), b.max())
    exportfile.write(w.T.astype('<f4').tobytes())
    exportfile.write(b.astype('<f4').tobytes())

    #mlpfinal
    w = model.value_linearfinal.weight.data.cpu().numpy()
    b = model.value_linearfinal.bias.data.cpu().numpy()

    print("mlpfinal_w", w.shape, w.min(), w.mean(), w.max())
    print("mlpfinal_b", b.shape, b.min(), b.mean(), b.max())
    exportfile.write(w.T.astype('<f4').tobytes())
    exportfile.write(np.zeros(5, dtype='<f4').tobytes())  # padding: (5,)
    exportfile.write(b.astype('<f4').tobytes())
    exportfile.write(np.zeros(5, dtype='<f4').tobytes())  # padding: (5,)

    exportfile.close()

    #copy txt file
    if (args.copy):
        exportCopyDestPath = modeldir + '/' + str(totalstep) + '.txt'
        shutil.copy(exportPath, exportCopyDestPath)

    print("success")
