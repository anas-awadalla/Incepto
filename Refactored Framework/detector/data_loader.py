# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
import os
from glob import glob

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets.mHealthData import mHealthData
from datasets.MotionSenseData import MotionSenseData
from datasets.oodParkinsonsData import oodParkinsonsData
from datasets.skindataset import SkinDataset
from datasets.parkinsons_dataset import parkinsonsData


def get_mPower(batch_size, TF, data_root='../Evaluating Models/Data/mPower/', train=True, val=True,col=14, **kwargs):
    ds = []
    files = os.listdir("../../../data3/mPower/data")
    train, test = train_test_split(files, test_size=0.8)
    if train:
        training_set = parkinsonsData(train, col=col)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    
    if val:
        validation_set = parkinsonsData(test, col=col)
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)

    return ds

def get_mHealth(batch_size, TF, data_root='../Evaluating Models/Data/mHealth/', train=True, val=True, **kwargs):
    ds = []
    
    if train:
        training_set = mHealthData()
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    
    if val:
        validation_set = mHealthData()
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)

    return ds


def get_MotionSense(batch_size, TF, data_root='../Evaluating Models/Data/MotionSense/', train=True, val=True, **kwargs):
    ds = []
    
    if train:
        training_set = MotionSenseData()
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        validation_set = MotionSenseData()
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)

    return ds


def get_oodParkinsons(batch_size, TF, data_root='../Evaluating Models/Data/oodParkinsons/', train=True, val=True, **kwargs):
    ds = []
    
    if train:
        training_set = oodParkinsonsData()
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    
    if val:
        validation_set = oodParkinsonsData()
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)

    return ds



def getHAM10000(batch_size, TF, data_root='../Evaluating Models/Data/skin-cancer-mnist-ham10000/', train=True, val=True, **kwargs):
    all_image_path = glob(os.path.join(data_root, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    
    df_original = pd.read_csv(os.path.join(data_root, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    
    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    df_undup.head()
    
    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    unique_list = list(df_undup['lesion_id'])
    df_original['duplicates'] = df_original['duplicates'].apply(lambda x: 'unduplicated' if x in unique_list else 'duplicated')
    
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
    
    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    val_list = list(df_val['image_id'])
    df_original['train_or_val'] = df_original['train_or_val'].apply(lambda x: 'val' if str(x) in val_list else 'train')
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']
    
    data_aug_rate = [15,10,5,50,0,5,40]
    for i in range(7):
        if data_aug_rate[i]:
            df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
            
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    
    ds = []
    
    if train:
        training_set = SkinDataset(df_train, transform=TF)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        validation_set = SkinDataset(df_val, transform=TF)
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)

    return ds

def getISIC2017(batch_size, TF, data_root='../Evaluating Models/Data/isic-2017/', train=True, val=True, **kwargs):
    ds = []
    
    if train:
        isic_2017_dir = data_root+'/ISIC-2017_Training_Data/'
        #isic_2017_df = pd.read_csv('../Evaluating Models/Data/isic-2017/ISIC-2017_Training_Part3_GroundTruth.csv')
        isic_2017_df = pd.read_csv(data_root+'/ISIC-2017_Training_Part3_GroundTruth.csv')
        isic_2017_df['path'] = isic_2017_df['image_id'].apply(lambda x: isic_2017_dir+x+'.jpg')
        cell_type_idx = []

        for i, row in isic_2017_df.iterrows():
            if row['melanoma'] == 0 and row['seborrheic_keratosis'] == 0:
                cell_type_idx.append(5)
            elif row['melanoma'] == 1:
                cell_type_idx.append(4)
            elif row['seborrheic_keratosis'] == 1:
                cell_type_idx.append(2)

        isic_2017_df['cell_type_idx'] = cell_type_idx
        train_loader = torch.utils.data.DataLoader(SkinDataset(isic_2017_df,transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
        
        ds.append(train_loader)
        
    if val:
        isic_2017_dir = data_root+'/ISIC-2017_Test_v2_Data/'
        #isic_2017_df = pd.read_csv('../Evaluating Models/Data/isic-2017/ISIC-2017_Training_Part3_GroundTruth.csv')
        isic_2017_df = pd.read_csv(data_root+'/ISIC-2017_Test_v2_Part3_GroundTruth.csv')
        isic_2017_df['path'] = isic_2017_df['image_id'].apply(lambda x: isic_2017_dir+x+'.jpg')
        cell_type_idx = []

        for i, row in isic_2017_df.iterrows():
            if row['melanoma'] == 0 and row['seborrheic_keratosis'] == 0:
                cell_type_idx.append(5)
            elif row['melanoma'] == 1:
                cell_type_idx.append(4)
            elif row['seborrheic_keratosis'] == 1:
                cell_type_idx.append(2)

        isic_2017_df['cell_type_idx'] = cell_type_idx
        test_loader = torch.utils.data.DataLoader(SkinDataset(isic_2017_df,transform=TF),batch_size=batch_size,shuffle=True, **kwargs)       
        
        ds.append(test_loader)
    
    return ds

def getHAM10000Transformed(batch_size, TF, tf_type='all', train=True, val=True, **kwargs):
    if tf_type == 'all':
        data_root='../Evaluating Models/Data/Ham10000Transformed/'
    else:
        data_root='../Evaluating Models/Data/Ham10000Transformed-ood/'+tf_type+'/'
        
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_root,transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
        ds.append(train_loader)
        
    if val:
        test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_root,transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
        ds.append(test_loader)
        
    return ds
    

def getISIC2016(batch_size, TF, data_root='../Evaluating Models/Data/skin-cancer-malignant-vs-benign/data/', train=True, val=True, **kwargs):
    ds = []
    
    if train:
        train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_root+'/train/',transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
        ds.append(train_loader)
        
    if val:
        test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_root+'/test/',transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
        ds.append(test_loader)
        
    return ds

def getFace(batch_size, TF, data_root='../Evaluating Models/Data/face/', train=True, val=True, **kwargs):
    ds = []
    train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_root,transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_root,transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
    
    
    ds.append(train_loader)
    ds.append(test_loader)
    
    return ds

def getFaceAge(batch_size, TF, data_root='../Evaluating Models/Data/face_age/', train=True, val=True, **kwargs):
    ds = []
    train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_root,transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_root,transform=TF),batch_size=batch_size,shuffle=True, **kwargs)
    
    
    ds.append(train_loader)
    ds.append(test_loader)
    
    return ds

def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds



def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'ham10000':
        train_loader, test_loader = getHAM10000(batch_size=batch_size, TF=input_TF, num_workers=1)
    elif data_type == 'mpower-rest':
        train_loader, test_loader = get_mPower(batch_size=batch_size, TF=input_TF, num_workers=1, col=14)
    elif data_type == 'mpower-outbound':
        train_loader, test_loader = get_mPower(batch_size=batch_size, TF=input_TF, num_workers=1, col=8)
    else: #data_type == 'mpower-return':
        train_loader, test_loader = get_mPower(batch_size=batch_size, TF=input_TF, num_workers=1, col=11)

    return train_loader, test_loader

def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'face':
        _, test_loader = getFace(batch_size=batch_size, TF=input_TF, num_workers=1)
    elif data_type == 'face_age':
        _, test_loader = getFaceAge(batch_size=batch_size, TF=input_TF, num_workers=1)
    elif data_type == 'isic-2016':
        _, test_loader = getISIC2016(batch_size=batch_size, TF=input_TF, num_workers=1)
    elif data_type == 'isic-2017':
        _, test_loader = getISIC2017(batch_size=batch_size, TF=input_TF, num_workers=1)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'ham10000-transformed':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='all', num_workers=2)
    elif data_type == 'ham10000-avg-smoothing':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='Avg Smoothing', num_workers=2)
    elif data_type == 'ham10000-brightness':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='Bright', num_workers=2)
    elif data_type == 'ham10000-contrast':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='Contrast', num_workers=2)
    elif data_type == 'ham10000-dilation':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='Dilation', num_workers=2)
    elif data_type == 'ham10000-erosion':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='Erosion', num_workers=2)
    elif data_type == 'ham10000-med-smoothing':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='Median Smoothing', num_workers=2)
    elif data_type == 'ham10000-rotation':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='Rotation', num_workers=2)
    elif data_type == 'ham10000-shift':
        _, test_loader = getHAM10000Transformed(batch_size=batch_size, TF=input_TF, tf_type='Shift', num_workers=2)
    elif data_type == 'MotionSense':
        _,test_loader = get_MotionSense(batch_size=batch_size, TF=input_TF, num_workers=1)
    elif data_type == 'mHealth':
        _,test_loader = get_mHealth(batch_size=batch_size, TF=input_TF, num_workers=1)
    elif data_type == 'mpower-rest':
        _, test_loader = get_mPower(batch_size=batch_size, TF=input_TF, num_workers=1, col=14)
    elif data_type == 'mpower-outbound':
        _, test_loader = get_mPower(batch_size=batch_size, TF=input_TF, num_workers=1, col=8)
    elif data_type == 'mpower-return':
         _, test_loader = get_mPower(batch_size=batch_size, TF=input_TF, num_workers=1, col=11)
    else: #data_type == 'oodParkinsons':
        _,test_loader = get_oodParkinsons(batch_size=batch_size, TF=input_TF, num_workers=1)
    
    
    return test_loader


