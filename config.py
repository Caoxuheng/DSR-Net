import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path',type=str,default='E:\Multispectral Image Dataset\CAVE\Mat_dataset/', help='where you store your HSI data file')
parser.add_argument('--srf_path',type=str,default='Nikon_D700.mat', help='where you save your HSI reconstruction results')
args=parser.parse_args()
