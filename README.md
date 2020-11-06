# Awesome-GANS-and-Deepfakes
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
A curated list of GAN & Deepfake papers and repositories. 

## GANs
- Vanilla GAN [(paper)](https://arxiv.org/abs/1406.2661)
    * Implementations
        * PyTorch 
            * [https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/gan]
            * [https://github.com/devnag/pytorch-generative-adversarial-networks]
            * [https://github.com/shivakanthsujit/GAN-PyTorch]
        * Keras
            * [https://github.com/eriklindernoren/Keras-GAN/tree/master/gan]
            * [https://github.com/osh/KerasGAN]
            * [https://github.com/jason71995/Keras-GAN-Library]
            * [https://github.com/Zackory/Keras-MNIST-GAN]
- DCGAN [(paper)](https://arxiv.org/abs/1511.06434)
    * Implementations
        * PyTorch
            * [https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN]
            * [https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/dcgan]
        * Keras 
            * [https://github.com/jacobgil/keras-dcgan]
            * [https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py]
            * [https://github.com/mitchelljy/DCGAN-Keras]
            
## Applications using GANs

### Anime generator
+ Towards the Automatic Anime Characters Creation with Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1708.05509)
+ [Project] Keras-GAN-Animeface-Character, [[github]](https://github.com/forcecore/Keras-GAN-Animeface-Character)

### Interactive Image generation
+ Generative Visual Manipulation on the Natural Image Manifold, [[paper]](https://arxiv.org/pdf/1609.03552), [[github]](https://github.com/junyanz/iGAN)
+ Neural Photo Editing with Introspective Adversarial Networks, [[paper]](http://arxiv.org/abs/1609.07093), [[github]](https://github.com/ajbrock/Neural-Photo-Editor)

### 3D Object generation
+ 3D Shape Induction from 2D Views of Multiple Objects, [[paper]](https://arxiv.org/pdf/1612.05872.pdf)
+ Parametric 3D Exploration with Stacked Adversarial Networks, [[github]](https://github.com/maxorange/pix2vox), [[youtube]](https://www.youtube.com/watch?v=ITATOXVvWEM)
+ Fully Convolutional Refined Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes, [[github]](https://github.com/yunishi3/3D-FCR-alphaGAN), [[blog]](https://becominghuman.ai/3d-multi-object-gan-7b7cee4abf80)

### Super-resolution
+ Image super-resolution through deep learning, [[github]](https://github.com/david-gpu/srez)
+ Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, [[paper]](https://arxiv.org/abs/1609.04802), [[github]](https://github.com/leehomyc/Photo-Realistic-Super-Resoluton)
+ High-Quality Face Image Super-Resolution Using Conditional Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1707.00737.pdf)
+ Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network, [[paper]](https://arxiv.org/pdf/1811.00344.pdf), [[github]](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw)

### Image Inpainting (hole filling)
+ Context Encoders: Feature Learning by Inpainting, [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf), [[github]](https://github.com/pathak22/context-encoder)
+ Semantic Image Inpainting with Perceptual and Contextual Losses, [[paper]](https://arxiv.org/abs/1607.07539), [[github]](https://github.com/bamos/dcgan-completion.tensorflow)
+ Generative Face Completion, [[paper]](https://drive.google.com/file/d/0B8_MZ8a8aoSeenVrYkpCdnFRVms/edit), [[github]](https://github.com/Yijunmaverick/GenerativeFaceCompletion)

### Image Segmentation
+ Vox2Vox: 3D-GAN for Brain Tumor Segmentation, [[paper]](https://arxiv.org/abs/2003.13653), [[github]](https://github.com/enochkan/vox2vox)

## Deepfakes

### Face-swapping
+ Fast Face-swap Using Convolutional Neural Networks, [[paper]](https://arxiv.org/abs/1611.09577), [[github]](https://github.com/deepfakes/faceswap#overview)

+ DeepFaceLab: A simple, flexible and extensible face
swapping framework, [[paper]](https://arxiv.org/pdf/2005.05535v4.pdf), [[github]](https://github.com/iperov/DeepFaceLab)

## Deepfake Detection

### CNN-based methods
+ MesoNet [[paper]](https://arxiv.org/abs/1809.00888), [[github]](https://github.com/HongguLiu/MesoNet-Pytorch)
+ Detecting Deep-Fake Videos from Phoneme-Viseme Mismatches, [[paper]](https://www.ohadf.com/papers/AgarwalFaridFriedAgrawala_CVPRW2020.pdf)
+ Deep Fake Image Detection Based on Pairwise Learning, [[paper]](https://www.mdpi.com/2076-3417/10/1/370)


 
 ## Datasets
+ [Google Deepfake Detection Dataset](https://github.com/ondyari/FaceForensics/tree/master/dataset)
+ [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics/tree/master/dataset)
+ [Facebook Deepfake Detection Challenge (DFDC) Dataset](https://www.kaggle.com/c/deepfake-detection-challenge/data)
+ ["SwapMe and Faceswap" dataset](https://www.sciencedirect.com/science/article/pii/S0957417419302350?via%3Dihub)
+ ["Fake Faces in the Wild (FFW) dataset](http://ali.khodabakhsh.org/research/ffw/)
+ [Tampered Face (TAMFA) Dataset](https://www.sciencedirect.com/science/article/pii/S0957417419302350?via%3Dihub)
+ [Celeb-DF(v2) Celebrity Deepfake Dataset](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)
+ [DeeperForensics-1.0](https://arxiv.org/pdf/2001.03024.pdf)
+ [Diverse Fake Face Dataset (DFFD)](https://arxiv.org/pdf/1910.01717.pdf)