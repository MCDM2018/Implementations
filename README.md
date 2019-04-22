# Implementations
Pytorch Implementation for

' Is Generator Conditioning Causally related to GAN performance? '

https://icml.cc/Conferences/2018/Schedule?showEvent=2868

How to run this ?

python main.py --dataset cifar10 --epoch 250 --eps 1 --eig_min 1 --eig_max 20 --result_dir generated_images --save_dir loss_and_weights

How to generate images with your trained weights?

python main.py --gen_mode True --result_dir To_be_generated_images_address --save_dir pretrained_weight_path

Requirements
- pytorch 0.4.0+
- python 3.6x
- cuda 9.0+
