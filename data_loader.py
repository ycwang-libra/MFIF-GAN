from torch.utils import data
from torchvision import transforms as T
from glob import glob
from PIL import Image
import os

class alpha_matte_AB(data.Dataset):
    """ trainning Dataset -- the alpha_matte_AB dataset."""

    def __init__(self, root_train, transform, transform2):

        self.transform = transform
        self.transform2 = transform2
        self.A_files = glob(os.path.join(root_train, 'A_jpg','*.*'))
        self.B_files = glob(os.path.join(root_train, 'B_jpg','*.*'))
        self.focus_map = glob(os.path.join(root_train, 'focus_map_png','*.*'))

    def __getitem__(self, index):

        A = Image.open(self.A_files[index])
        B = Image.open(self.B_files[index])
        focus_map = Image.open(self.focus_map[index])
        
        A = self.transform(A)
        B = self.transform(B)
        fm = self.transform2(focus_map).split(1,0)[0]
        
        return A, B, fm

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)
    
class LytroDataset(data.Dataset):
    """test Dataset -- the LytroDataset."""
    def __init__(self, root_test, transform):
        self.transform = transform
        self.A_files = glob(os.path.join(root_test, 'Lytro/LytroDataset/A_jpg','*.*'))
        self.B_files = glob(os.path.join(root_test, 'Lytro/LytroDataset/B_jpg','*.*'))

    def __getitem__(self, index):

        A = Image.open(self.A_files[index])
        B = Image.open(self.B_files[index])
        A = self.transform(A)
        B = self.transform(B)
        
        return A, B

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)

class MFFW2(data.Dataset):
    """test Dataset -- the MFFW2 Dataset."""
    def __init__(self, root_test, transform):
        self.transform = transform
        self.A_files = glob(os.path.join(root_test, 'MFFW/MFFW2/A_jpg','*.*'))
        self.B_files = glob(os.path.join(root_test, 'MFFW/MFFW2/B_jpg','*.*'))

    def __getitem__(self, index):

        A = Image.open(self.A_files[index])
        B = Image.open(self.B_files[index])
        A = self.transform(A)
        B = self.transform(B)
        
        return A, B

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)
    
class grayscale_jpg(data.Dataset):
    """test Dataset -- the MFFW2 Dataset."""
    def __init__(self, root_test, transform):
        self.transform = transform
        self.A_files = glob(os.path.join(root_test, 'grayscale/A_jpg','*.*'))
        self.B_files = glob(os.path.join(root_test, 'grayscale/B_jpg','*.*'))

    def __getitem__(self, index):

        A = Image.open(self.A_files[index])
        B = Image.open(self.B_files[index])
        A = self.transform(A)
        B = self.transform(B)
        
        return A, B

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)

def get_loader(root_train, root_test, crop_size, image_size, 
               batch_size,  mode, test_dataset, current_model_name, num_workers):
    """Build and return a data loader."""
    transform = []
    transform2 = []
    if mode == 'train':
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        
        transform2.append(T.CenterCrop(crop_size))
        transform2.append(T.Resize(image_size))
        transform2.append(T.ToTensor())
        transform2.append(T.Lambda(lambda x: x.repeat(3,1,1)))
        transform2.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform2 = T.Compose(transform2)
    else:
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        
        transform2.append(T.ToTensor())
        transform2.append(T.Lambda(lambda x: x.repeat(3,1,1)))
        transform2.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform2 = T.Compose(transform2)
    

    if mode == 'train':
        dataset = alpha_matte_AB(root_train, transform, transform2)    #groundtruth, A, B
    else:
        if test_dataset == 'Lytro':
            dataset = LytroDataset(root_test, transform)       #A, B
        elif test_dataset == 'MFFW2':
            dataset = MFFW2(root_test, transform)
        elif test_dataset == 'grayscale_jpg':
            dataset = grayscale_jpg(root_test, transform2)
        
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader