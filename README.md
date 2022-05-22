### FFUNet-PyTorch

This is a PyTorch implementation for FFUNet image denoising, as in Wencong Wu and Yungang Zhang. "FFUNet: a fast and flexible U-shaped network for image denoising."

### How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [HDF5 for Python](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### 2. data preparation

### 2.1 color image
        python prepare_patches.py

### 2.2 grayscale image
        python prepare_patches.py --gray

**NOTES**  *--max_number_patches* can be used to set the maximum number of patches contained in the database. *--aug_times* can be used to set the number of data augmentation, we set it as 5.

### 3. Train FFUNet

### 3.1 grayscale image
```
python train_gray.py --batch_size 128 --epochs 80 --noiseIntL 0 75 --val_noiseL 25
```

### 3.2 color image
```
python train_rgb.py --batch_size 128 --epochs 80 --noiseIntL 0 75 --val_noiseL 25
```

### 4. Test FFUNet

### 4.1 grayscale image
```
python avg_gray_test.py --input Set12 --noise_sigma 25 --add_noise True
```

To run the algorithm on CPU instead of GPU:
```
python avg_gray_test.py --input Set12 --noise_sigma 25 --add_noise True --no_gpu
```

### 4.2 color image
```
python avg_rgb_test.py --input Kodak24 --noise_sigma 25 --add_noise True
```

To run the algorithm on CPU instead of GPU:
```
python avg_rgb_test.py --input Kodak24 --noise_sigma 25 --add_noise True --no_gpu
```
