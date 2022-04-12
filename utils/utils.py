import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
from config import *
import torch
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor
import pdb
import seaborn as sns
from sklearn import metrics

def buildDataLoaders_denseNet(data_augmentation=False):
    # Transform to resize data to DenseNet dimensions
    basic_transforms =  transforms.Compose([
        transforms.Resize(DENSE_NET_INPUT_SIZE), 
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not data_augmentation:
        transform = basic_transforms
    else:
        transform=transforms.Compose([
            transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.75,1.25), shear=15),
            transforms.ColorJitter(brightness=(0.2,0.8), contrast=(0.2, 0.8)),
            basic_transforms
        ])

    trainset = datasets.MNIST(DATA_DIR, download=True, train=True, transform=transform)
    validationset = datasets.MNIST(DATA_DIR, download=True, train=False, transform=basic_transforms)
    # Build loaders
    train_loader = DataLoader(trainset, batch_size=BACTH_SIZE, shuffle=True)
    val_loader = DataLoader(validationset, batch_size=BACTH_SIZE, shuffle=True)
    dataloaders_dict = {}
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['val'] = val_loader

    return dataloaders_dict

def buildDataLoaders_UNET(image_size):

    trainset = SegmentationMNIST(DATA_DIR, train=True, image_size=image_size)
    validationset = SegmentationMNIST(DATA_DIR, train=False, image_size=image_size)
    # imshow(trainset)
    # pdb.set_trace()
    # Build loaders
    train_loader = DataLoader(trainset, batch_size=BACTH_SIZE2, shuffle=True)
    val_loader = DataLoader(validationset, batch_size=BACTH_SIZE2, shuffle=True)
    dataloaders_dict = {}
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['val'] = val_loader

    return dataloaders_dict


def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    since = time.time()

    for epoch in range(num_epochs):
        init_epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
        print(f"Time per epoch: {time.time() - init_epoch_time}")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model

def train_unet(model, dataloaders, criterion, optimizer, num_epochs=5):
    since = time.time()

    for epoch in range(num_epochs):
        init_epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device=DEVICE, dtype=torch.float32)
            labels = labels.to(device=DEVICE, dtype=torch.long).squeeze(1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels) 
                loss += dice_loss(F.softmax(outputs, dim=1), F.one_hot(labels, N_CLASES_UNET).permute(0, 3, 1, 2).float(), multiclass=True)
                #_, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            #running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print('{} Loss: {:.4f}'.format('train', epoch_loss))
        print(f"Time per epoch: {time.time() - init_epoch_time}")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model


def eval_model(model, testloader, criterion):
    since = time.time()
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in testloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(testloader.dataset)
    epoch_acc = running_corrects.double() / len(testloader.dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def eval_unet(model, testloader, n_samples):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    actual_samples = 0

    # METRICS
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Batch is of size 1
    for inputs, labels in testloader:
        # Batch is of size 1
        assert inputs.shape[0] == 1

        inputs = inputs.to(device=DEVICE, dtype=torch.float32)
        labels = labels.to(device=DEVICE, dtype=torch.long).squeeze(1)

        # Para mostrar imagen
        if actual_samples < n_samples:
            plt.clf()
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(labels.permute(1, 2, 0).cpu())
            axs[0].set_title('True mask')

        labels = F.one_hot(labels, N_CLASES_UNET).permute(0, 3, 1, 2).float()
        
        with torch.no_grad():
            # predict the mask
            output = model(inputs)
            image_mask = output.argmax(dim=1)
            if actual_samples < n_samples:
                axs[1].imshow(image_mask.permute(1, 2, 0).cpu())
                axs[1].set_title('Predicted mask')
                actual_samples += 1
                plt.savefig(MODEL_SAVE_DIR + "_" + str(actual_samples) + ".png")
            output = F.one_hot(image_mask, N_CLASES_UNET).permute(0, 3, 1, 2).float()

            # Metricas
            FN += (output - labels == -1).sum() # FN
            FP += (output - labels == 1).sum()  # FP
            TP += (output + labels == 2).sum()  # TP
            TN += (output + labels == 0).sum()  # TN

    # MATRIZ CONFUSION GENERAL
    cmat = [[TP.cpu(), FN.cpu()], [FP.cpu(), TN.cpu()]]
    plt.clf()
    sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(MODEL_SAVE_DIR + "_confusionMatrix.png")
    
    # CURVAS ROC
    # Por hacer

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# SHOW DATASET
def imshow(dataloader):
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    #pdb.set_trace()
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)



class SegmentationMNIST(Dataset):
    def __init__(self, root:str, train:bool, image_size=(128,128), transform=None, target_transform = None, ndigits = (5,8), max_iou=0.1):
        """
        Args:
        - root: the route for the MNIST dataset. It will be downloaded if it does not exist
        - train: True for training set, False for test set
        - image_size: tuple with the dataset image size
        - transform: the transforms to be applied to the input image
        - target_transform: the transforms to be applied to the label image
        - ndigits: tuple with the mininum and maximum number of digits per image
        - max_iou: maximum IOU between digit bounding boxes
        """

        self.transform = transform
        self.target_transform = target_transform
        self.image_template = torch.zeros((1, *image_size), dtype=torch.float32)
        self.target_tempalte = torch.zeros((1, *image_size), dtype=torch.uint8)
        self.max_iou = 0.2

        mnist_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.5,1.5), shear=30),
            transforms.Resize(48),
            transforms.ToTensor()
        ])

        self.mnist = datasets.MNIST(root, train, download=True, transform = mnist_transform)
        self.index = random.permutation(len(self.mnist))

        # Compute the number of digits in each image, and the total number of images
        self.num_digits = []
        remaining = len(self.mnist)
        while remaining > ndigits[1]+ndigits[0]:  # The remaining will be from min to max, i.e. one image
            this_num = random.randint(ndigits[0], ndigits[1])
            self.num_digits.append(this_num)
            remaining -= this_num
        self.num_digits.append(remaining)

        self.num_digits = np.array(self.num_digits)
        self.start_digit = self.num_digits.cumsum() - self.num_digits[0]

    def __len__(self):
        return len(self.num_digits)
    
    def __getitem__(self, idx):
        sample = self.image_template.detach().clone()
        target = self.image_template.detach().clone()

        for i in range(self.num_digits[idx]):
            # if self.start_digit[idx]+i == 60000:
            #     pdb.set_trace()
            #digit, cls = self.mnist[self.start_digit[idx]+i]   # esto da error
            digit, cls = self.mnist[(self.start_digit[idx]+i) % len(self.mnist)]   # parche tmeporal
            mask = digit>0

            valid = False
            while not valid: 
                y = random.randint(0, sample.size(1)-digit.size(1))
                x = random.randint(0, sample.size(2)-digit.size(2))
                valid = (mask*sample[:, y:y+digit.size(1), x:x+digit.size(2)]>0).sum() < self.max_iou*mask.sum()

            sample[:,y:y+digit.size(1), x:x+digit.size(2)][mask] = digit[mask]
            target[:,y:y+digit.size(1), x:x+digit.size(2)][mask] = cls+1

            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return (sample, target)