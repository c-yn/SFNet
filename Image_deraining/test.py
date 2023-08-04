import os
import torch
import argparse
from models.SFNet import build_net
from data import test_dataloader
from utils import Adder
import time
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as f

class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'input/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

def test_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(path, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


parser = argparse.ArgumentParser()

# Directories
parser.add_argument('--model_name', default='SFNet', type=str)
parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/deraining_testset')

parser.add_argument('--test_model', type=str, default='/root/autodl-tmp/sfnet/deraining.pkl')
parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
args = parser.parse_args()

args.result_dir = os.path.join('results/', args.model_name, 'deraining/')

if not os.path.exists('results/'):
    os.makedirs(args.model_save_dir)
if not os.path.exists('results/' + args.model_name + '/'):
    os.makedirs('results/' + args.model_name + '/')
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

model = build_net()

if torch.cuda.is_available():
    model.cuda()

state_dict = torch.load(args.test_model)
model.load_state_dict(state_dict['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
adder = Adder()
model.eval()

datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']

for dataset in datasets:
    if not os.path.exists(args.result_dir+dataset+'/'):
        os.makedirs(args.result_dir+dataset)
    print(args.result_dir+dataset)
    dataloader = test_dataloader(os.path.join(args.data_dir, dataset), batch_size=1, num_workers=4)
    factor = 8
    with torch.no_grad():
        psnr_adder = Adder()


        # Main Evaluation
        for iter_idx, data in enumerate(tqdm(dataloader), 0):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')


            tm = time.time()

            pred = model(input_img)[2]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                save_name = os.path.join(args.result_dir, dataset, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

        print('==========================================================')
