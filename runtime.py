from trainer import Trainer
import argparse
from PIL import Image
import os
def main(lr=0.0002,l1_coef=50,l2_coef=50,diter=5,cls_m=False,vis_screen='gan',save_path='/home/soham19477/EEE511ANC-Text2ImageGAN-master/EEE511ANC-Text2ImageGAN-master/PyTorch/data/flowers/netG_affine/checks1',inference=0,pre_trained_disc=None,pre_trained_gen=None,split=0,batch_size=64,num_workers=8,epochs=51,photos_path="test_images1"):

    trainer = Trainer(split=split,
                      lr=lr,
                      diter=diter,
                      vis_screen=vis_screen,
                      save_path=save_path,
                      l1_coef=l1_coef,
                      l2_coef=l2_coef,
                      pre_trained_disc=pre_trained_disc,
                      pre_trained_gen=pre_trained_gen,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      epochs=epochs,
                      photos_path=photos_path
                      )
    
    if inference==0:
        print("OK")
        trainer.train(cls_m)
    elif inference==1:
        trainer.predict()
    elif inference==2:
        trainer.predict2()
    else:
        trainer.get_IS()
# main(photos_path="test_images1",split=2,cls_m=True,inference=3,pre_trained_disc="data/flowers/netG_affine/checks1/disc_50.pth",pre_trained_gen="data/flowers/netG_affine/checks1/gen_50.pth")
main(cls_m=True,save_path="/home/soham19477/EEE511ANC-Text2ImageGAN-master/EEE511ANC-Text2ImageGAN-master/PyTorch/data/flowers/netG_affine_first_half/checks1",epochs=51)
# 3.1651305085601726 for 100, 2.975432156746411 for 50 affine
#