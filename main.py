import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
        
    if not os.path.exists(config.loss_dir):
        os.makedirs(config.loss_dir)
    if not os.path.exists(config.loss_graph_dir):
        os.makedirs(config.loss_graph_dir)

    # if not os.path.exists(config.focus_map_dir):
    #     os.makedirs(config.focus_map_dir)
    # if not os.path.exists(config.focus_map_pp_dir):
    #     os.makedirs(config.focus_map_pp_dir)   
    
    result_dir = os.path.join(config.result_root_dir, config.current_model_name)   

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Data loader.
    train_loader = None
    test_loader = None

    if config.mode == 'train':
        print('^^^^^^ train ^^^^')
        train_loader = get_loader(config.root_train, config.root_test, config.crop_size, config.image_size, 
                                   config.batch_size, config.mode, config.test_dataset, config.current_model_name, config.num_workers)
    else:
        print('^^^^^^ test ^^^^^^')
        test_loader = get_loader(config.root_train, config.root_test, config.crop_size, config.image_size, 
                                   config.batch_size, config.mode, config.test_dataset, config.current_model_name, config.num_workers)

    # Solver for training and testing MFIF-GAN.
    solver = Solver(train_loader, test_loader, config)
    
    if config.mode == 'train':
        if config.dataset in ['alpha_matte_AB']:
            solver.train()

    elif config.mode == 'test':
        if config.dataset in ['alpha_matte_AB']:
            solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model name
    parser.add_argument('--model_name', type=str, default='MFIF_GAN')

    # Model configuration.
    parser.add_argument('--model', type=str, default=None, help='the type of model', choices=['Gen', 'Recon'])
    parser.add_argument('--crop_size', type=int, default=178, help='crop size for the alpha_matte_AB dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=9, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=3, help='number of strided conv layers in D')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='alpha_matte_AB', choices=['alpha_matte_AB', 'Both'])
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size') # Select the appropriate batch size according to the GPU memory
    parser.add_argument('--num_iters', type=int, default=120000, help='number of total iterations for training D')  
    parser.add_argument('--num_iters_decay', type=int, default=10000, help='number of iterations for decaying lr') 
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')  
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')


    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=110000, help='test model from this step') 
    parser.add_argument('--current_model_name', type=str, default='9_basic_big1_pp0_001_11w')
    parser.add_argument('--test_dataset', type=str, default='Lytro', 
                        choices=['Lytro','MFFW2','grayscale_jpg'])

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0) # win
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # Directories.
    parser.add_argument('--root_train', type=str, default='~/dataset')
    parser.add_argument('--root_test', type=str, default='~/dataset')
    parser.add_argument('--log_dir', type=str, default='MFIF_GAN/logs')
    parser.add_argument('--model_save_dir', type=str, default='MFIF_GAN/models')
    parser.add_argument('--sample_dir', type=str, default='MFIF_GAN/samples')
    parser.add_argument('--loss_dir', type=str, default='MFIF_GAN/Loss')
    parser.add_argument('--loss_graph_dir', type=str, default='MFIF_GAN/Loss_graph')
    
    # parser.add_argument('--focus_map_dir', type=str, default='MFIF_GAN/results_Lytro_focus_map_11w')
    # parser.add_argument('--focus_map_pp_dir', type=str, default='MFIF_GAN/results_Lytro_focus_map_pp0_001_11w')

    parser.add_argument('--result_root_dir', type=str, default='MFIF_GAN/Fusion_result')
   
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
    