import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from models_gray import FEUNet
from utils import batch_psnr, normalize, variable_to_cv2_image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parse arguments
parser = argparse.ArgumentParser(description="FEUNet_Test")
parser.add_argument('--add_noise', type=str, default="True")
parser.add_argument("--input", type=str, default="Set12", help='path to input image ')
parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
args = parser.parse_args()
# Normalize noises ot [0, 1]
args.noise_sigma /= 255.
# use CUDA?
args.cuda = not args.no_gpu and torch.cuda.is_available()


def main():
    print('Loading model ...\n')
    # Create model
    net = FEUNet(num_input_channels=1)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join('models', 'net_gray.pth')))
    model.eval()

    files_source = glob.glob(os.path.join('data', 'test', args.input, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        # from HxWxC to  CxHxW grayscale image (C=1)
        imorig = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        imorig = np.expand_dims(imorig, 0)
        imorig = np.expand_dims(imorig, 0)

        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = imorig.shape
        if sh_im[2] % 2 == 1:
            expanded_h = True
            imorig = np.concatenate((imorig, imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

        if sh_im[3] % 2 == 1:
            expanded_w = True
            imorig = np.concatenate((imorig, imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

        imorig = normalize(imorig)
        imorig = torch.Tensor(imorig)

        # Sets data type according to CPU or GPU modes
        if args.cuda:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # Add noise
        if args.add_noise:
            noise = torch.FloatTensor(imorig.size()).normal_(mean=0, std=args.noise_sigma)
            imnoisy = imorig + noise
        else:
            imnoisy = imorig.clone()

        # Test mode
        with torch.no_grad():
            imorig, imnoisy = Variable(imorig.type(dtype)), Variable(imnoisy.type(dtype))
            nsigma = Variable(torch.FloatTensor([args.noise_sigma]).type(dtype))

        # Estimate noise and subtract it to the input image
        im_noise_estim = model(imnoisy, nsigma)
        outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)

        if expanded_h:
            imorig = imorig[:, :, :-1, :]
            outim = outim[:, :, :-1, :]
            imnoisy = imnoisy[:, :, :-1, :]

        if expanded_w:
            imorig = imorig[:, :, :, :-1]
            outim = outim[:, :, :, :-1]
            imnoisy = imnoisy[:, :, :, :-1]

        outimg = variable_to_cv2_image(outim)
        cv2.imwrite("FEUNet.png", outimg)

        if args.add_noise:
            psnr = batch_psnr(outim, imorig, 1.)
            psnr_test += psnr
            print("%s PSNR %.2f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)


if __name__ == "__main__":
    main()

