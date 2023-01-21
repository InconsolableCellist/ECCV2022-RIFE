#!/usr/bin/python3

import os
import cv2
import torch
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

args = {}
args['modelDir'] = 'train_log'
args['exp'] = 2

try:
    from train_log.RIFE_HDv3 import Model
    model = Model()
    model.load_model(args['modelDir'], -1)
    print("Loaded v3.x HD model.")
except Exception as e:
    print(f"Failed to load v3.x HD model: {e}")
    os.sys.exit(1)

model.eval()
model.device()

def gen_frame(img0_name, img1_name):
    img0 = cv2.imread(os.path.join('input_images', img0_name), cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(os.path.join('input_images', img1_name), cv2.IMREAD_UNCHANGED)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_list = [img0, img1]
    for i in range(args['exp']):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

    for i in range(0, len(img_list) - 2):
        cv2.imwrite(f'output/{img0_name[:-4]}.{i}.png', (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])


if not os.path.exists('output'):
    os.mkdir('output')

images = [f for f in os.listdir('input_images') if f.endswith('.png')]
images.sort()
for i in range(0, len(images), 1):
    if os.path.exists('output/' + images[i][:-4] + '.0.png'):
        print('s', end='')
        continue

    gen_frame(images[i], images[i+1])



