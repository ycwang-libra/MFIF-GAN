import torch 
from torchvision import transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_same_size(A, focus_map):
    '''
        Input: aim_size_image
               Image_need_to_be_resize
        Output: resized_image
    '''
    A_size = list(A.size())
    focus_map = focus_map.squeeze(dim = 0).squeeze(dim = 1).cpu()
    focus_map = T.ToPILImage()(focus_map)
    crop_obt = T.CenterCrop((A_size[2],A_size[3]))
    focus_map = crop_obt(focus_map)
    focus_map = T.ToTensor()(focus_map)
    focus_map = focus_map.unsqueeze(dim = 0).to(device)
    return focus_map
