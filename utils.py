import numpy as np
import scipy.io as sio
import imgvision as iv
import torch

def get_load_attr(path:str):
    if '.npy' in path:
        load_attr = np.load
    elif '.mat' in path:
        load_attr = sio.loadmat
    return load_attr

def load_HSI(args,idx,dataset:str='CAVE'):
    name = dataset.lower()
    path = args.data_path+f'  ({idx}).mat'
    loader = get_load_attr(path)
    if name =='cave':
        HSI = loader(path)
    elif name =='harvard':
        HSI = loader(path)['ref']
        HSI = HSI/HSI.max()
        HSI = HSI[:1024,:1024]
    elif name =='icvl':
        import h5py
        HSI = np.array(h5py.File(path)['rad']).T
    try:
        HSI = HSI['HSI']
    except:
        pass
    return HSI

def preprocess_HSI(img,illuminant:str='D65'):
    SPD = iv.spectra(illuminant=illuminant).illuminance
    img /=SPD[0]
    return img/img.max()

def get_degradation(illuminant:str,R,case=' '):
    L = iv.spectra(illuminant=illuminant).illuminance

    L = np.diag(L[:,0])
    L /=L.max()
    M = L @ R


    if case =='ls':
        return L
    return M

def dual_LC_img(HSI,R,case=2):
    M65 = get_degradation('D65',R)
    MA = get_degradation('A',R)
    Y_65 =  HSI @ M65
    Y_A =  HSI@ MA
    if case ==3:
        MC = get_degradation('C',R)
        Y_C = HSI@MC
        return Y_65, Y_A,Y_C

    return Y_65,Y_A

def tensor2mx(data):
    return data[0].detach().cpu().numpy().T

def mx2tensor(data):
    return torch.tensor(data.T).float().unsqueeze(0)

def HSI2MSI(x,srf):
    return (x[0].permute(1,2,0) @srf) .permute(2,0,1).unsqueeze(0)

def decompose_rgb_structure(img,S):
    R,G,B = [img[:,:,i] for i in range(3)]

    print(R.shape)
def SAM_GPU(output, label):
        ratio = (torch.sum((output + 1e-8).multiply(label + 1e-8), axis=1)) / (torch.sqrt(
            torch.sum((output + 1e-8).multiply(output + 1e-8), axis=1) * torch.sum(
                (label + 1e-8).multiply(label + 1e-8), axis=1)))
        angle = torch.acos(ratio.clip(-1, 1))

        return torch.mean(angle)
def PSNR_GPU(im_true, im_fake):
    data_range = 1
    _,C,H,W = im_true.size()
    err = torch.pow(im_true.clone()-im_fake.clone(),2).mean(dim=(-1,-2), keepdim=True)
    psnr = 10. * torch.log10((data_range**2)/err)
    return torch.mean(psnr)
def fspecial_gauss(size, sigma):
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=0)
    x_data = np.expand_dims(x_data, axis=0)

    y_data = np.expand_dims(y_data, axis=0)
    y_data = np.expand_dims(y_data, axis=0)

    x = torch.tensor(x_data).float()
    y = torch.tensor(y_data).float()

    g = torch.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / torch.sum(g)

def SSIM_LOSS(img1, img2, size=11, sigma=1.5 ):
    # window shape [size, size]
    window= fspecial_gauss(size, sigma).cuda()
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = torch.nn.functional.conv2d(img1, window, padding='same')
    mu2 =  torch.nn.functional.conv2d(img2, window,padding='same')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq =  torch.nn.functional.conv2d(img1*img1, window,padding='same') - mu1_sq
    sigma2_sq =  torch.nn.functional.conv2d(img2*img2, window,padding='same') - mu2_sq
    sigma12 =  torch.nn.functional.conv2d(img1*img2, window, padding='same') - mu1_mu2

    v1 = 2*mu1_mu2+C1
    v2 = mu1_sq+mu2_sq+C1

    value = (v1*(2.0*sigma12 + C2))/(v2*(sigma1_sq + sigma2_sq + C2))
    value = torch.mean(value)
    value = 1.0-value
    return value





if __name__ == '__main__':
    from config import args
    import matplotlib.pyplot as plt
    HSI = load_HSI(args,1)
    HSI = preprocess_HSI(HSI)
    S = sio.loadmat(args.srf_path)['R']
    MA1,LA = get_degradation('D65',S,case='ls')
    MA2 = get_degradation('A', S)
    print(MA1-MA2)
    plt.imshow(HSI@LA@S)
    plt.show()






