import numpy as np
from regex import R
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from txt2image_dataset import Text2ImageDataset
from models.gan_factory import gan_factory
from utils import Utils, Logger
from PIL import Image
import os
from transform import Encode

from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

from scipy.stats import entropy

# print(yaml.__version__)
class Trainer(object):
    def __init__(self, split, lr, diter, vis_screen, save_path, l1_coef, l2_coef, pre_trained_gen, pre_trained_disc, batch_size, num_workers, epochs, photos_path):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f,yaml.FullLoader)
        self.photos_path=photos_path
        self.generator = torch.nn.DataParallel(gan_factory.generator_factory().cuda())
        self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory().cuda())

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        print('Loading Flowers')
        self.dataset = Text2ImageDataset(config['flowers_dataset_path'], split=split)
        self.v_dataset = Text2ImageDataset(config['flowers_dataset_path'], split=1)
        print('Finish Loading')
        
        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.DITER = diter

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)
        
        self.v_data_loader = DataLoader(self.v_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.logger = Logger(vis_screen)
        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path

    def train(self, cls=False):
        self._train_gan(cls)
        
    def _train_gan(self, cls):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0

        for epoch in range(self.num_epochs):
            d=0
            g=0
            r=0
            f=0
            self.generator.train()
            self.discriminator.train()
            for sample in self.data_loader:
                iteration += 1
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                wrong_images = sample['wrong_images']

                right_images = Variable(right_images.float()).cuda()
                right_embed = Variable(right_embed.float()).cuda()
                wrong_images = Variable(wrong_images.float()).cuda()

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss
                if cls:
                    d_loss = d_loss + wrong_loss
                
                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)


                #======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                #===========================================

                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)
                
                g_loss.backward()
                self.optimG.step()

                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch,d_loss, g_loss, real_score, fake_score, self.save_path)
                    self.logger.draw(right_images, fake_images)
                g+=g_loss.data.cpu().mean().item()
                d+=d_loss.data.cpu().mean().item()
                r+=real_score.data.cpu().mean().item()
                f+=fake_score.data.cpu().mean().item()
                torch.cuda.empty_cache()
            d/=len(self.data_loader)
            g/=len(self.data_loader)
            r/=len(self.data_loader)
            f/=len(self.data_loader)
            print(len(self.data_loader))
            self.logger.plot_epoch_w_scores(iteration)
            df = open("d.txt","a")
            df.write(str(d)+"\n")
            df.close()
            df = open("g.txt","a")
            # f = open(path+"/g.txt", "a")
            df.write(str(g)+"\n")
            df.close()
            df = open("dx.txt","a")
            # f = open(path+"/dx.txt", "a")
            df.write(str(r)+"\n")
            df.close()
            df = open("dgx.txt","a")
            # f = open(path+"/dgx.txt", "a")
            df.write(str(f)+"\n")
            df.close()
            # self.logger.plot_epoch_w_scores(epoch)

            self.generator.eval()
            self.discriminator.eval()
            d=0
            g=0
            r=0
            f=0
            with torch.no_grad():
                for sample in self.v_data_loader:
                    iteration += 1
                    right_images = sample['right_images']
                    right_embed = sample['right_embed']
                    wrong_images = sample['wrong_images']

                    right_images = Variable(right_images.float()).cuda()
                    right_embed = Variable(right_embed.float()).cuda()
                    wrong_images = Variable(wrong_images.float()).cuda()

                    real_labels = torch.ones(right_images.size(0))
                    fake_labels = torch.zeros(right_images.size(0))

                    # ======== One sided label smoothing ==========
                    # Helps preventing the discriminator from overpowering the
                    # generator adding penalty when the discriminator is too confident
                    # =============================================
                    smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                    real_labels = Variable(real_labels).cuda()
                    smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                    fake_labels = Variable(fake_labels).cuda()

                    # Train the discriminator
                    self.discriminator.zero_grad()
                    outputs, activation_real = self.discriminator(right_images, right_embed)
                    real_loss = criterion(outputs, smoothed_real_labels)
                    real_score = outputs

                    if cls:
                        outputs, _ = self.discriminator(wrong_images, right_embed)
                        wrong_loss = criterion(outputs, fake_labels)
                        wrong_score = outputs

                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                    noise = noise.view(noise.size(0), 100, 1, 1)
                    fake_images = self.generator(right_embed, noise)
                    outputs, _ = self.discriminator(fake_images, right_embed)
                    fake_loss = criterion(outputs, fake_labels)
                    fake_score = outputs

                    d_loss = real_loss + fake_loss

                    if cls:
                        d_loss = d_loss + wrong_loss

                    
                    # Train the generator
                    self.generator.zero_grad()
                    noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                    noise = noise.view(noise.size(0), 100, 1, 1)
                    fake_images = self.generator(right_embed, noise)
                    outputs, activation_fake = self.discriminator(fake_images, right_embed)
                    _, activation_real = self.discriminator(right_images, right_embed)

                    activation_fake = torch.mean(activation_fake, 0)
                    activation_real = torch.mean(activation_real, 0)


                    #======= Generator Loss function============
                    # This is a customized loss function, the first term is the regular cross entropy loss
                    # The second term is feature matching loss, this measure the distance between the real and generated
                    # images statistics by comparing intermediate layers activations
                    # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                    # because it links the embedding feature vector directly to certain pixel values.
                    #===========================================
                    # print("fr",fake_images.shape,right_images.shape)
                    g_loss = criterion(outputs, real_labels) \
                            + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                            + self.l1_coef * l1_loss(fake_images, right_images)
                    g+=g_loss.data.cpu().mean().item()
                    d+=d_loss.data.cpu().mean().item()
                    r+=real_score.data.cpu().mean().item()
                    f+=fake_score.data.cpu().mean().item()
                    torch.cuda.empty_cache()
            d/=len(self.v_data_loader)
            g/=len(self.v_data_loader)
            r/=len(self.v_data_loader)
            f/=len(self.v_data_loader)
            print(len(self.v_data_loader))
            # self.logger.plot_epoch_w_scores(iteration)
            df = open("vd.txt","a")
            df.write(str(d)+"\n")
            df.close()
            df = open("vg.txt","a")
            # f = open(path+"/g.txt", "a")
            df.write(str(g)+"\n")
            df.close()
            df = open("vdx.txt","a")
            # f = open(path+"/dx.txt", "a")
            df.write(str(r)+"\n")
            df.close()
            df = open("vdgx.txt","a")
            # f = open(path+"/dgx.txt", "a")
            df.write(str(f)+"\n")
            df.close()

            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)

    def predict(self):
        for sample in self.data_loader:
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            txt = sample['txt']
            # requires /data/flowers/netG_affine/results to exist
            if not os.path.exists('./data/flowers/netG_affine/results/{0}'.format(self.photos_path)):
                os.makedirs('./data/flowers/netG_affine/results/{0}'.format(self.photos_path))

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()

            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)

            self.logger.draw(right_images, fake_images)

            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('./data/flowers/netG_affine/results/{0}/{1}.jpg'.format(self.photos_path, t.replace("/", "").replace(".","").replace("\n","").replace(",","")[:100]))
                # print(t)

    def get_IS(self):
        iteration = 0
        sis=0
        with torch.no_grad():
            for sample in self.data_loader:
                iteration += 1
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                wrong_images = sample['wrong_images']

                right_images = Variable(right_images.float()).cuda()
                right_embed = Variable(right_embed.float()).cuda()
                wrong_images = Variable(wrong_images.float()).cuda()

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                sis+=self.inception_score(fake_images)[0]
                torch.cuda.empty_cache()
        print(sis/len(self.v_data_loader))
        return sis/len(self.v_data_loader)

    def predict2(self):

        f = open("demofile.txt", "r")
        x = f.read()
        txt = x.split('\n')
        print(txt)
        txt+=[" "]
        right_embed = Encode(txt)

        # if not os.path.exists('results/{0}'.format(self.save_path)):
        #     os.makedirs('results/{0}'.format(self.save_path))

        right_embed = right_embed.numpy()
        right_embed = torch.from_numpy(right_embed)
        right_embed = Variable(right_embed)

        # Train the generator
        noise = Variable(torch.randn(len(txt), 100))
        noise = noise.view(noise.size(0), 100, 1, 1)
        fake_images = self.generator(right_embed, noise)

        # require save_path/results_inference to exist
        for image, t in zip(fake_images, txt):
            im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('{0}/results_inference/{1}.jpg'.format(self.save_path, t.replace("/", "").replace(".","").replace("\n","").replace(",","")[:100]))
            print(t)



# Sourced from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

    def inception_score(self,imgs,cuda=True, batch_size=64, resize=True, splits=1):
        """Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
        """
        N = len(imgs)
        # print()
        batch_size=min(batch_size,N-1)
        assert batch_size > 0
        assert N > batch_size

        # Set up dtype
        if cuda:
            dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            dtype = torch.FloatTensor

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model.eval()
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
        def get_pred(x):
            if resize:
                x = up(x)
            x = inception_model(x)
            print(x.shape)
            return F.softmax(x,dim=1).data.cpu().numpy()

        # Get predictions
        preds = np.zeros((N, 1000))

        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)



