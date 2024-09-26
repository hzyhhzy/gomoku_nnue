
from model import Model_v3,Model_v3a
from model import boardH,boardW
from collections import OrderedDict
import argparse
import numpy as np
import torch
import os
import time
import shutil

try:
    os.mkdir("../export")
except:
    pass
else:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str ,default='fs40b_v3_c128', help='model path')
    parser.add_argument('--export', type=str ,default='fs40b_v3a_c128', help='export path')
    parser.add_argument('--mlpc2', type=int ,default=32, help='mid mlp output')

    args = parser.parse_args()

    device=torch.device('cpu')
    modelname=args.model
    exportname=args.export
    if(exportname==''):
        exportname=modelname+"_v3a"






    file_path = f'../saved_models/{modelname}/model.pth'
    model_type=None
    if os.path.exists(file_path):
        data = torch.load(file_path, map_location=device)
        model_type = data['model_type']
        model_param = data['model_param']
        if(model_type != "v3"):
            print(f"Invalid Model Type: {model_type}")
            exit(0)
        model = Model_v3(*model_param).to(device)

        model.load_state_dict(data['state_dict'])
        totalstep = data['totalstep']
        print(f"loaded model: type={model_type}, param={model.model_param}, totalstep={totalstep}")
    else:
        print(f"Invalid Model Path: {file_path}")
        exit(0)

    basepath = f'../saved_models/{args.export}/'
    if not os.path.exists(basepath):
        os.mkdir(basepath)


    model.eval()

    # 初始化新模型
    mlpc2 = args.mlpc2  # 新参数
    v3a_model = Model_v3a(model_param[0],model_param[1],model_param[2],model_param[3],mlpc2,model_param[4])

    # 复制参数
    original_state_dict = model.state_dict()
    modified_state_dict = v3a_model.state_dict()


    # 创建一个新的OrderedDict来保存更新后的参数
    new_state_dict = OrderedDict()

    # 排除需要重新初始化的层
    excluded_layers = ['value_linear4', 'value_linearfinal', 'mlp_policy_w']

    for name, param in original_state_dict.items():
        if not any(excluded_layer in name for excluded_layer in excluded_layers):
            new_state_dict[name] = param

    # 将剩余的随机初始化层参数加入new_state_dict中
    for name, param in modified_state_dict.items():
        if name not in new_state_dict:
            new_state_dict[name] = param

    # 加载修改后的state_dict
    v3a_model.load_state_dict(new_state_dict)

    torch.save(
        {'totalstep': totalstep,
         'state_dict': v3a_model.state_dict(),
         'model_type': v3a_model.model_type,
         'model_param': v3a_model.model_param},
        basepath+"model.pth")



    print("success")






