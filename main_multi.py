import numpy as np
import os
from config import args
import matplotlib.pyplot as plt
import imgvision as iv
from Model.Unet import skip
from utils import *
import torch
from torch import  nn
import torchvision



def spectral_varation(img):
    a = img[:,1:]-img[:,:-1]
    return torch.abs(a[:,1:]-a[:,:-1]).mean()

class Zm_Bn(nn.Module):
    def __init__(self):
        super(Zm_Bn,self).__init__()
    def forward(self,x):
        return x - nn.AdaptiveAvgPool2d(1)(x)


class ModelBlock(nn.Module):
    def __init__(self,R,L65,LA,band=31):
        super(ModelBlock, self).__init__()
        self.R_ = R
        self.RT = nn.Sequential(nn.Conv2d(3,band,kernel_size=3,padding=1),
                                nn.Conv2d(band,band,kernel_size=1),nn.ReLU())
        self.L65 = L65
        self.LA = LA
        self.b=Zm_Bn()

    def R(self,HSI):
        return (HSI[0].permute(1, 2, 0) @ self.R_).permute(2, 0, 1).unsqueeze(0)

    def L(self,HSI,L_):
        return (HSI[0].permute(1, 2, 0) @ L_).permute(2, 0, 1).unsqueeze(0)

    def forward(self,x,Y_65,Y_A):
        Res65 = self.R(self.L(x,self.L65))-Y_65
        ResA = self.R(self.L(x, self.LA)) - Y_A
        Inv_HSI_65 = self.L(self.RT(Res65),self.L65)
        Inv_HSI_A = self.L(self.RT(ResA),self.LA)
        return x + self.b(0.5*(Inv_HSI_A+Inv_HSI_65))
class ModelBlock(nn.Module):
    def __init__(self,R,L65,LA,band=31):
        super(ModelBlock, self).__init__()
        self.R_ = R
        self.RT = nn.Sequential(nn.Conv2d(3,band,kernel_size=3,padding=1),
                                nn.Conv2d(band,band,kernel_size=1),nn.ReLU())
        self.L65 = L65
        self.LA = LA

    def R(self,HSI):
        return (HSI[0].permute(1, 2, 0) @ self.R_).permute(2, 0, 1).unsqueeze(0)

    def L(self,HSI,L_):
        return (HSI[0].permute(1, 2, 0) @ L_).permute(2, 0, 1).unsqueeze(0)

    def forward(self,x,Y_65,Y_A):
        Res65 = self.R(self.L(x,self.L65))-Y_65
        ResA = self.R(self.L(x, self.LA)) - Y_A
        Inv_HSI_65 = self.L(self.RT(Res65),self.L65)
        Inv_HSI_A = self.L(self.RT(ResA),self.LA)
        return x - 0.3*(Inv_HSI_A+Inv_HSI_65)

class MyNet(nn.Module):
    def __init__(self,R,L65,LA):
        super(MyNet, self).__init__()

        self.GSD1 = ModelBlock(R, L65, LA)
        self.GSD2 = ModelBlock(R, L65, LA)
        self.GSD3 = ModelBlock(R, L65, LA)
        n=4

        #
        self.prox1 = skip(31, 31,[40],[40],[1], n_scales=n)
        #
        self.prox2 = skip(31, 31,[40],[40],[1], n_scales=n)

        # self.prox3 = skip(31, 31,[40],[40],[1], n_scales=n)


        self.bn = Zm_Bn()



    def forward(self,x,Y_65,Y_A):

        x = self.prox1(self.GSD1(x, Y_65, Y_A))
        x = self.prox2(self.GSD2(x, Y_65, Y_A))
        x =self.GSD3(x, Y_65, Y_A)


        return x.clamp(0,1)

from ssim_torch import  SSIM
ssim_loss = SSIM()
for imgidx in range(1,32):
    S = sio.loadmat(args.srf_path)['R']
    HSI = load_HSI(args,imgidx,'cave')[:,:]
    HSI = preprocess_HSI(HSI)
    l65 = get_degradation('D65', S,case='ls')
    lA = get_degradation('a', S,case='ls')

    y_65 = HSI @ l65 @ S
    y_A = HSI @ lA @ S


    Y_65 = mx2tensor(y_65).cuda()
    Y_A = mx2tensor(y_A).cuda()
    LA = torch.FloatTensor(lA).cuda()
    L65 = torch.FloatTensor(l65).cuda()
    R = torch.FloatTensor(S).cuda()
    Net = MyNet(R,L65,LA).cuda()

    # from thop import profile

    model_folder='D:/Spectral super_resolution/Dual_Illuminances_Spectra_Recovery/Result/' + str(imgidx) + 'd565A'
    trainer = torch.optim.Adam(params=Net.parameters(),lr=5e-3)
    sche = torch.optim.lr_scheduler.StepLR(trainer,500,0.95)

    HSI_cuda = mx2tensor( HSI).cuda()
    PRE = torch.rand_like(HSI_cuda)
    L1 = torch.nn.L1Loss()

    for i in range(5001):
        trainer.zero_grad()
        recon = Net(PRE,Y_65,Y_A)

        Y_65_p,Y_A_p = HSI2MSI(recon,L65 @ R),HSI2MSI(recon,LA @ R)
        # Y_65_a, Y_A_a = HSI2MSI(a, L65 @ R), HSI2MSI(a, LA @ R)
        # Y_65_b, Y_A_b = HSI2MSI(b, L65 @ R), HSI2MSI(b, LA @ R)


        loss = L1(Y_A_p,Y_A)+L1(Y_65,Y_65_p)+0.3*spectral_varation(recon)
        loss.backward()
        trainer.step()
        sche.step()
        if i%10==0:
            print('\r{:.3f}\t{:.3f}\t{}'.format(PSNR_GPU(recon,HSI_cuda),SAM_GPU(recon,HSI_cuda)*180/torch.pi,i),end='')

    checkpoint = {
        "net": Net.state_dict(),
        'optimizer': trainer.state_dict(),

    }
    model_out_path = model_folder + "{}.pth".format(i)

    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    torch.save(checkpoint, model_out_path)



    recon = tensor2mx(recon)
    iv.spectra_metric(HSI, recon).Evaluation()
    np.save('D:/Spectral super_resolution/Dual_Illuminances_Spectra_Recovery/Result/'+str(imgidx)+'d565A',recon)
    #



# print(abs(rgb-RGB).mean(0).mean(0)*255)
# from PIL import Image
# for i in range(31):
#     im = Image.fromarray(np.uint8(HSI[:,:,i].clip(0,1)*255))
#     im.save(f'img/{i}.png')
# plt.imshow(iv.spectra().space(recon))
# plt.show()
while True:
    x,y = int(input('x')),int(input('y'))
    plt.plot(HSI[x,y],'g',recon[x,y],'b')
    plt.show()
    plt.imshow(iv.spectra().space(recon))
    plt.show()
