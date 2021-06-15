import torch 
from skimage.morphology import remove_small_objects
import numpy as np
def post_remove_small_objects(input_image, size=None):
    if size is None:
        _, _, H, W = input_image.shape
        size = 0.001*H*W
    if type(input_image) is torch.Tensor:
        ar = input_image.detach().cpu().numpy()
    ar[ar>0.5]=1
    ar[ar<=0.5]=0
    ar=ar.astype(np.bool)
    tmp_image1 = remove_small_objects(ar, size)
    tmp_image2 = (1-tmp_image1).astype(np.bool)
    tmp_image3 = remove_small_objects(tmp_image2, size)
    tmp_image4 = 1-tmp_image3
    tmp_image4 = tmp_image4.astype(np.float)

    if type(input_image) is torch.Tensor:
        tmp_image4 = torch.from_numpy(tmp_image4)
        tmp_image4 = tmp_image4.to(input_image.device)
    return tmp_image4