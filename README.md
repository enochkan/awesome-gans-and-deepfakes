# Awesome-GANS-and-Deepfakes
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)

A curated list of GAN & Deepfake papers and repositories. :heavy_check_mark: means implementation is available.

## GANs
Tl;dr GANs containg two competing neural networks which iteratively generate new data with the same statistics as the training set.

### Unconditional GANs
+ :heavy_check_mark: Vanilla GAN: Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1406.2661), [[github]](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/gan)
+ :heavy_check_mark: DCGAN: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1511.06434), [[github]](https://github.com/carpedm20/DCGAN-tensorflow)
+ :heavy_check_mark: WGAN: Wasserstein GAN, [[paper]](https://arxiv.org/abs/1701.07875), [[github]](https://github.com/martinarjovsky/WassersteinGAN)
+ :heavy_check_mark: WGAN-GP: Improved Training of Wasserstein GANs, [[paper]](https://arxiv.org/pdf/1704.00028.pdf), [[github]](https://github.com/caogang/wgan-gp)
+ :heavy_check_mark: RGAN: The relativistic discriminator: a key element missing from standard GAN, [[paper]](https://arxiv.org/abs/1807.00734), [[github]](https://github.com/AlexiaJM/RelativisticGAN)
+ :heavy_check_mark: BGAN: Boundary-Seeking Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1702.08431), [[github]](implementations/bgan/bgan.py)
+ :heavy_check_mark: ClusterGAN: Latent Space Clustering in Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1809.03627), [[github]](https://github.com/sudiptodip15/ClusterGAN)

### Conditional GANs
+ :heavy_check_mark: CGAN: Conditional Generative Adversarial Nets, [[paper]](https://arxiv.org/abs/1411.1784), [[github]](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py)
+ :heavy_check_mark: ACGAN: Conditional Image Synthesis With Auxiliary Classifier GANs, [[paper]](https://arxiv.org/abs/1610.09585), [[github]](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py)
+ :heavy_check_mark: CCGAN: Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1611.06430), [[github]](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/ccgan/ccgan.py)

### Image-to-Image Translation
+ :heavy_check_mark: CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, [[paper]](https://arxiv.org/abs/1703.10593), [[github]](https://github.com/junyanz/CycleGAN)
+ :heavy_check_mark: StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation, [[paper]](https://arxiv.org/abs/1711.09020), [[github]](https://github.com/yunjey/stargan)
+ :heavy_check_mark: Pix2Pix: Image-to-Image Translation with Conditional Adversarial Nets, [[paper]](https://arxiv.org/abs/1611.07004), [[github]](https://github.com/phillipi/pix2pix)
+ :heavy_check_mark: DualGAN: Unsupervised Dual Learning for Image-to-Image Translation, [[paper]](https://arxiv.org/abs/1704.02510), [[github]](https://github.com/duxingren14/DualGAN)
+ :heavy_check_mark: BicycleGAN: Toward Multimodal Image-to-Image Translation, [[paper]](https://arxiv.org/abs/1711.11586), [[github]](https://github.com/junyanz/BicycleGAN)

### Volumetric (3D) Generation
+ :heavy_check_mark: 3DGAN: Learning a Probabilistic Latent Space of Object Shapes
via 3D Generative-Adversarial Modeling, [[paper]](http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf), [[github]](https://github.com/enochkan/3dgan-keras)
+ :heavy_check_mark: Inverse Graphics GAN: Inverse Graphics GAN - Learning to Generate 3D Shapes from Unstructured 2D Data, [[paper]](https://arxiv.org/pdf/2002.12674.pdf), [[github]](https://github.com/BraneShop/showreel/issues/504)

## Applications using GANs

### Anime generator
+ Towards the Automatic Anime Characters Creation with Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1708.05509)
+ :heavy_check_mark: [Project] Keras-GAN-Animeface-Character, [[github]](https://github.com/forcecore/Keras-GAN-Animeface-Character)

### Interactive Image generation
+ :heavy_check_mark: Generative Visual Manipulation on the Natural Image Manifold, [[paper]](https://arxiv.org/pdf/1609.03552), [[github]](https://github.com/junyanz/iGAN)
+ :heavy_check_mark: Neural Photo Editing with Introspective Adversarial Networks, [[paper]](http://arxiv.org/abs/1609.07093), [[github]](https://github.com/ajbrock/Neural-Photo-Editor)

### 3D Object generation
+ 3D Shape Induction from 2D Views of Multiple Objects, [[paper]](https://arxiv.org/pdf/1612.05872.pdf)
+ :heavy_check_mark: Parametric 3D Exploration with Stacked Adversarial Networks, [[github]](https://github.com/maxorange/pix2vox), [[youtube]](https://www.youtube.com/watch?v=ITATOXVvWEM)
+ :heavy_check_mark: Fully Convolutional Refined Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes, [[github]](https://github.com/yunishi3/3D-FCR-alphaGAN), [[blog]](https://becominghuman.ai/3d-multi-object-gan-7b7cee4abf80)

### Super-resolution
+ :heavy_check_mark: Image super-resolution through deep learning, [[github]](https://github.com/david-gpu/srez)
+ :heavy_check_mark: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, [[paper]](https://arxiv.org/abs/1609.04802), [[github]](https://github.com/leehomyc/Photo-Realistic-Super-Resoluton)
+ High-Quality Face Image Super-Resolution Using Conditional Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1707.00737.pdf)
+ :heavy_check_mark: Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network, [[paper]](https://arxiv.org/pdf/1811.00344.pdf), [[github]](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw)
+ :heavy_check_mark: ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1809.00219), [[github]](https://github.com/xinntao/ESRGAN)
+ :heavy_check_mark: MUNIT: Multimodal Unsupervised Image-to-Image Translation, [[paper]](https://arxiv.org/abs/1804.04732), [[github]](https://github.com/nvlabs/MUNIT)
+ :heavy_check_mark: SRGAN: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, [[paper]](https://arxiv.org/abs/1609.04802), [[github]](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py)

### Image Inpainting (hole filling)
+ :heavy_check_mark: Context Encoders: Feature Learning by Inpainting, [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf), [[github]](https://github.com/pathak22/context-encoder)
+ :heavy_check_mark: Semantic Image Inpainting with Perceptual and Contextual Losses, [[paper]](https://arxiv.org/abs/1607.07539), [[github]](https://github.com/bamos/dcgan-completion.tensorflow)
+ :heavy_check_mark: Generative Face Completion, [[paper]](https://drive.google.com/file/d/0B8_MZ8a8aoSeenVrYkpCdnFRVms/edit), [[github]](https://github.com/Yijunmaverick/GenerativeFaceCompletion)

### Medical Image Segmentation
+ :heavy_check_mark: Vox2Vox: 3D-GAN for Brain Tumor Segmentation, [[paper]](https://arxiv.org/abs/2003.13653), [[github]](https://github.com/enochkan/vox2vox)
+ SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation, [[paper]](https://arxiv.org/abs/1706.01805)
+ Generative Adversarial Neural Networks for Pigmented and Non-Pigmented Skin Lesions Detection in Clinical Images, [[paper]](https://ieeexplore.ieee.org/document/7968584/)

## Deepfakes
Tl;dr Deepfakes are fake videos or audio recordings that look and sound just like the real thing. Watch this video of [Obama speaking](https://www.youtube.com/watch?v=cQ54GDm1eL0)... or was that really him?

### CNN-based Face-swapping
+ :heavy_check_mark: Fast Face-swap Using Convolutional Neural Networks, [[paper]](https://arxiv.org/abs/1611.09577), [[github]](https://github.com/deepfakes/faceswap#overview)
+ :heavy_check_mark: DeepFaceLab: A simple, flexible and extensible face
swapping framework, [[paper]](https://arxiv.org/pdf/2005.05535v4.pdf), [[github]](https://github.com/iperov/DeepFaceLab)

### GAN-based Face-swapping
+ :heavy_check_mark: Fewshot Face Translation GAN, [[github]](https://github.com/shaoanlu/fewshot-face-translation-GAN)
+ Faceswap-GAN, [[github]](https://github.com/shaoanlu/faceswap-GAN)
+ :heavy_check_mark: AttGAN: Facial Attribute Editing by Only
Changing What You Want, [[paper]](http://vipl.ict.ac.cn/uploadfile/upload/2019112511573287.pdf), [[github]](https://github.com/LynnHo/AttGAN-Tensorflow)
+ MulGAN: Facial Attribute Editing by Exemplar, [[paper]](https://arxiv.org/abs/1912.12396)
+ :heavy_check_mark: MaskGAN: Towards Diverse and Interactive Facial Image Manipulation, [[paper]](https://arxiv.org/abs/1907.11922), [[github]](https://github.com/switchablenorms/CelebAMask-HQ)
+ :heavy_check_mark: StarGAN v2: Diverse Image Synthesis for Multiple Domains, [[paper]](https://arxiv.org/abs/1912.01865), [[github]](https://github.com/clovaai/stargan-v2)
+ :heavy_check_mark: FSGAN: Subject Agnostic Face Swapping and Reenactment, [[paper]](https://arxiv.org/pdf/1908.05932.pdf), [[github]](https://github.com/YuvalNirkin/fsgan)

## Deepfake Detection

### CNN-based methods
+ :heavy_check_mark: MesoNet [[paper]](https://arxiv.org/abs/1809.00888), [[github]](https://github.com/HongguLiu/MesoNet-Pytorch)
+ Detecting Deep-Fake Videos from Phoneme-Viseme Mismatches, [[paper]](https://www.ohadf.com/papers/AgarwalFaridFriedAgrawala_CVPRW2020.pdf)
+ Deep Fake Image Detection Based on Pairwise Learning, [[paper]](https://www.mdpi.com/2076-3417/10/1/370)

### RCN-based methods
+ Recurrent Convolutional Strategies for Face Manipulation Detection in Videos, [[paper]](https://arxiv.org/pdf/1905.00582.pdf)

### Other ML methods
+ SVM: Exposing Deep Fakes Using Inconsistent Head Poses, [[paper]](https://ieeexplore.ieee.org/document/8683164)
 
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
