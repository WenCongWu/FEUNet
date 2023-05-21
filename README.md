### FEUNet-PyTorch

This is a PyTorch implementation for FEUNet image denoising. Paper download: * [FEUNet](https://link.springer.com/article/10.1007/s11760-022-02471-1)

### How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [HDF5 for Python](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### 2. Data Preparation

### 2.1 dataset download

* [The Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/)

### 2.2 color image
        python prepare_patches.py --p 50

### 2.3 grayscale image
        python prepare_patches.py --p 64 --gray

**NOTES**  *--max_number_patches* can be used to set the maximum number of patches contained in the database. *--aug_times* can be used to set the number of data augmentation, we set it as 5.

### 3. Train FEUNet

### 3.1 grayscale image
```
python train_gray.py --batch_size 128 --epochs 80 --noiseIntL 0 75 --val_noiseL 25
```

### 3.2 color image
```
python train_rgb.py --batch_size 128 --epochs 80 --noiseIntL 0 75 --val_noiseL 25
```

### 4. Test FEUNet

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

### 5. Network Structure

![FEUNet](https://user-images.githubusercontent.com/106000336/223597505-6ac56131-c1cf-4a48-a8b5-96eb8223a61c.png)

### 6. Results

### 6.1 gaussian grayscale image denoising

![image](https://user-images.githubusercontent.com/106000336/223598361-1f0f5788-bbc6-461d-878f-d50335ef3ac9.png)

![image](https://user-images.githubusercontent.com/106000336/223598430-2f115d3b-7dd7-434a-989d-deee4aefb0ec.png)

![image](https://user-images.githubusercontent.com/106000336/223598612-f5449490-9fc7-4371-aac2-fe38a47dc9e8.png)

### 6.2 gaussian color image denoising

![image](https://user-images.githubusercontent.com/106000336/223598674-2d307ee4-0fb4-4062-a478-c2ce80dc359c.png)

### 6.3 real image denoising

![image](https://user-images.githubusercontent.com/106000336/223598926-3f6c0fc9-2340-47d3-867a-f462264214b8.png)

### 6.4 image smoothing

![image](https://user-images.githubusercontent.com/106000336/223598967-54fda992-e7af-4796-aa70-f05b8e675b54.png)

![image](https://user-images.githubusercontent.com/106000336/223599031-34e2c68b-5224-4528-87f8-a62551913e89.png)

### 6.5 model computational complexity comparison

![image](https://user-images.githubusercontent.com/106000336/223599159-455e4de5-4ffd-4a16-bd53-32d9d010cd8b.png)

![image](https://user-images.githubusercontent.com/106000336/223599243-4129aeca-88dc-498b-8cec-845f76455c99.png)

### 7. Citation
```
@article{wu2023feunet,
  title={FEUNet: a flexible and effective U-shaped network for image denoising},
  author={Wu, Wencong and Lv, Guannan and Liao, Shicheng and Zhang, Yungang},
  journal={Signal, Image and Video Processing},
  pages={1--9},
  year={2023}
}
```
### 8. Other Important Notes

Some codes are from [An Analysis and Implementation of the FFDNet Image Denoising Method](http://www.ipol.im/pub/art/2019/231/)


