from model import Generator
from model import Discriminator
from torchvision.utils import save_image
import torch
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
from CNN_post_process import post_remove_small_objects
from to_same_size import to_same_size
import skimage.io

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

class Solver(object):
    """Solver for training and testing MFIF-GAN."""

    def __init__(self, alpha_loader, lytro_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.alpha_loader = alpha_loader
        self.lytro_loader = lytro_loader
        
        # Model name
        self.model_name = config.model_name
        self.current_model_name = config.current_model_name
        # Model configurations.
        self.model = config.model
        self.test_dataset = config.test_dataset
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.loss_dir = config.loss_dir
        self.loss_graph_dir = config.loss_graph_dir
        
        # New Directories
        # self.focus_map_dir = config.focus_map_dir
        # self.focus_map_pp_dir = config.focus_map_pp_dir           
        self.result_root_dir = config.result_root_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['alpha_matte_AB']:
            self.G = Generator(self.g_conv_dim, self.g_repeat_num, self.model)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator, reconstructor and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
                        
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):      
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):
        """Train MFIF-GAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'alpha_matte_AB':
            data_loader = self.alpha_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        A_fixed, B_fixed, focus_map_fixed = next(data_iter)    
        focus_map_fixed = focus_map_fixed.to(self.device)
        A_fixed = A_fixed.to(self.device)
        B_fixed = B_fixed.to(self.device)
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
            
            loss_d_path = os.path.join(self.loss_dir, 'Loss_D_{}.npy'.format(self.resume_iters))
            loss_g_path = os.path.join(self.loss_dir, 'Loss_G_{}.npy'.format(self.resume_iters))

        # Start training.
        print('Start training...', self.model_name)
        start_time = time.time()
        
        if self.resume_iters:
            GG_Loss = torch.from_numpy(np.load(loss_g_path)).to(self.device)
            DD_Loss = torch.from_numpy(np.load(loss_d_path)).to(self.device)
            G_Loss = torch.zeros(self.resume_iters // self.n_critic + 1).to(self.device)  #reset a new empty tensor (size: self.resume_iters / self.n_critic + 1) 
            D_Loss = torch.zeros(self.resume_iters // self.n_critic + 1).to(self.device)
            G_Loss[1:] = GG_Loss    # input G_Loss behind the second 
            D_Loss[1:] = DD_Loss
        else:
            G_Loss = torch.zeros(1).to(self.device)
            D_Loss= torch.zeros(1).to(self.device)
            
        average_num = 200    #number of average section
        
        #########################################
        j = start_iters // self.n_critic     # Loss number      
        for i in range(start_iters, self.num_iters):  # iteration of training

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                A, B, focus_map = next(data_iter)   
            except:
                data_iter = iter(data_loader)
                A, B, focus_map= next(data_iter)

            focus_map = focus_map.to(self.device)
            A = A.to(self.device)
            B = B.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with ground truth images.
            d_real_decision = self.D.train()(A, B, focus_map)
            d_loss_real = - torch.mean(d_real_decision)   

            # Compute loss with fake images.
            decision_map = self.G.train()(A, B)
            d_fake_decision = self.D.train()(A.detach(), B.detach(), decision_map.detach())    
            d_loss_fake = torch.mean(d_fake_decision)    

            # Compute loss for gradient penalty.
            alpha = torch.rand(focus_map.size(0), 1, 1, 1).to(self.device)
            img_hat = (alpha * focus_map.data + (1 - alpha) * decision_map.data).requires_grad_(True)
            d_hat_decision = self.D.train()(A, B, img_hat)
            d_loss_gp = self.gradient_penalty(d_hat_decision, img_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
  
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D_Loss'] = d_loss.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:  #n_critic = 5   number of D updates per each G update is 5
                j += 1
                
                G_Loss_New = torch.zeros(j + 1).to(self.device)
                D_Loss_New = torch.zeros(j + 1).to(self.device)
                
                decision_map = self.G.train()(A, B)
                d_fake_decision = self.D.train()(A, B, decision_map)
                g_loss_fake = - torch.mean(d_fake_decision)    

                g_loss_rec = torch.mean(torch.abs(focus_map - decision_map))  

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec
                
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                
                # ========================== save Loss
                D_Loss_New[:j ] = D_Loss
                D_Loss_New[j ] = d_loss.detach()
                D_Loss = D_Loss_New
                
                del D_Loss_New
                
                G_Loss_New[:j ] = G_Loss
                G_Loss_New[j ] = g_loss.detach()
                G_Loss = G_Loss_New     #  G loss (first value is 0)
                
                del G_Loss_New

                # ===============  Logging.
                loss['G_Loss'] = g_loss.item()

            # =================================================================================== #
            #                                 5. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:  #log_step = 10
                used_time = time.time() - start_time
                remain_time = used_time / (i + 1 - start_iters) * (self.num_iters - start_iters) - used_time
                et = str(datetime.timedelta(seconds = used_time))[:-7]
                et2 = str(datetime.timedelta(seconds = remain_time))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}], Remain time [{}]".format(et, i+1, self.num_iters, et2)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:  #sample_step = 1000 
                with torch.no_grad():
                    ''' orders: A, B, focus_map, decision_map, A1, B1    '''
                    
                    img_list = [A_fixed]
                    img_list.append(B_fixed)
                    img_list.append(focus_map_fixed.repeat(1,3,1,1))
                    decision_map = self.G(A_fixed, B_fixed)
                    img_list.append(decision_map.repeat(1,3,1,1))
                    img_concat = torch.cat(img_list, dim = 3)

                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(img_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fused images into {}...'.format(sample_path))
                    print('This training model is ...', self.model_name)
                    
            # save loss graph     
            if (i+1) % self.sample_step == 0:  #sample_step = 1000 
                    
               #save Loss
                G_real_Loss = G_Loss[1:]
                D_real_Loss = D_Loss[1:]
                g_loss = G_real_Loss.cpu().numpy()
                d_loss = D_real_Loss.cpu().numpy()
                loss_d_path = os.path.join(self.loss_dir, 'Loss_D_{}'.format(j*5))
                loss_g_path = os.path.join(self.loss_dir, 'Loss_G_{}'.format(j*5))
                np.save(loss_d_path,d_loss)
                np.save(loss_g_path,g_loss)
                
                # loss graph
                interval_len = j//average_num   # length of average section
                x = [ interval_len * self.n_critic * n for n in range(j//interval_len)]    
                G_aver_loss = np.zeros(average_num)
                D_aver_loss = np.zeros(average_num)
                G_split = list_split(g_loss, interval_len)  # cut to (average_num) parts
                D_split = list_split(d_loss, interval_len)
                
                for idx in range(average_num):
                    G_aver_loss[idx] = np.mean(G_split[idx])
                    D_aver_loss[idx] = np.mean(D_split[idx])
                
                plt.clf()
                plt.plot(x,G_aver_loss)
                plt.plot(x,D_aver_loss)
                plt.legend(['G_loss','D_Loss'])
                Loss_path = os.path.join(self.loss_graph_dir, 'Loss_{}.png'.format(j*self.n_critic))
                plt.savefig(Loss_path)
                # plt.show()

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:  # model_save_step = 10000
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.    #lr_update_step = 1000    num_iters_decay = 25000
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):  
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using MFIF-GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)   #test_iters  = 50000 help='test model from this step'
        
        # Set data loader.            
        if self.dataset == 'alpha_matte_AB':
            data_loader = self.lytro_loader  #test data from LytroDataset output are A, B 

        
        with torch.no_grad():
            fusion_start_time = time.time()
            for i, (A, B) in enumerate(data_loader):

                '''
                # test show origin image、focus_map、fused_img
                A = A.to(self.device)
                B = B.to(self.device)

                decision_map = self.G.eval()(A, B)
                focus_map = post_remove_small_objects(decision_map).float()
                fused_img = A * focus_map + B * (1 - focus_map)
                
                img_list = [A]
                img_list.append(B)
                img_list.append(decision_map.repeat(1,3,1,1))
                img_list.append(focus_map.repeat(1,3,1,1))
                img_list.append(fused_img)
                
                img_concat = torch.cat(img_list, dim=3)
                
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(img_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
                '''
                
                # '''
                #testing save fused_img
                A = A.to(self.device)
                B = B.to(self.device)

                decision_map = self.G.eval()(A, B)

                if decision_map.size(2) != A.size(2) or decision_map.size(3) != A.size(3):
                    decision_map = to_same_size(A, decision_map)
                    
                focus_map = post_remove_small_objects(decision_map).float()
                fused_img = A * focus_map + B * (1 - focus_map)
                
                fusion_used_time = time.time() - fusion_start_time
                
                if self.test_dataset == 'Lytro':
                    result_path = os.path.join(self.result_root_dir,self.current_model_name, 'Lytro_fusion_{}.jpg'.format(i+1))
                elif self.test_dataset == 'MFFW2':
                    result_path = os.path.join(self.result_root_dir,self.current_model_name, 'MFFW2_fusion_{}.jpg'.format(i+1))
                elif self.test_dataset == 'grayscale_jpg':
                    result_path = os.path.join(self.result_root_dir,self.current_model_name, 'grayscale_fusion_{}.jpg'.format(i+1))
                
            
                if self.test_dataset == 'grayscale_jpg':
                    # save one channel gray image
                    fused_img = fused_img.split(1,dim = 1)[0]
                    fused_img = torch.squeeze(fused_img)
                    fused_img = self.denorm(fused_img.data.cpu())
                    fused_img = fused_img * 255
                    fused_img = fused_img.numpy().astype('uint8')
                    skimage.io.imsave(result_path, fused_img)
                else:
                    #save three channels image
                    save_image(self.denorm(fused_img.data.cpu()), result_path, nrow=1, padding=0) 
                        

                print('Saved fused image into {}...'.format(result_path))
                print('Used time is {}'.format(fusion_used_time))
                # '''
                
                '''
                #testing save foreground
                A = A.to(self.device)
                B = B.to(self.device)

                decision_map = self.G.eval()(A, B)
                focus_map = post_remove_small_objects(decision_map).float()
#                fused_img = A * focus_map + B * (1 - focus_map)
                foreground_img = A * focus_map
                
                fusion_used_time = time.time() - fusion_start_time     # time time time time time time time 
                
                # result_path = os.path.join(self.result_dir, 'Lytro_MFIF-GAN_foreground_{}.jpg'.format(i+1))
                result_path = os.path.join(self.result_root_dir, 'Lytro_MFIF-GAN0_001_foreground_{}.jpg'.format(i+1))
#                save_image(self.denorm(fused_img.data.cpu()), result_path, nrow=1, padding=0)
                save_image(self.denorm(foreground_img.data.cpu()), result_path, nrow=1, padding=0)


                print('Saved focus map into {}...'.format(result_path))
                print('Used time is {}'.format(fusion_used_time))     # time time time time time time time 
                '''
                
                '''
                #testing save background
                A = A.to(self.device)
                B = B.to(self.device)

                decision_map = self.G.eval()(A, B)
                focus_map = post_remove_small_objects(decision_map).float()
#                fused_img = A * focus_map + B * (1 - focus_map)
                background_img = B * (1 - focus_map)
                
                fusion_used_time = time.time() - fusion_start_time
                
                # result_path = os.path.join(self.result_dir, 'Lytro_MFIF-GAN_background_{}.jpg'.format(i+1))
                result_path = os.path.join(self.result_root_dir, 'Lytro_MFIF-GAN0_001_background_{}.jpg'.format(i+1))
                save_image(self.denorm(background_img.data.cpu()), result_path, nrow=1, padding=0)
#                skimage.io.imsave(result_path,background_img.data.cpu())


                print('Saved focus map into {}...'.format(result_path))
                print('Used time is {}'.format(fusion_used_time))
                '''
                
                '''
                #testing save fused_img no_pp
                A = A.to(self.device)
                B = B.to(self.device)

                focus_map = self.G.eval()(A, B)
                fused_img = A * focus_map + B * (1 - focus_map)
                
                fusion_used_time = time.time() - fusion_start_time
                
                if self.test_dataset == 'Lytro':
                    result_path = os.path.join(self.result_dir, 'Lytro_fusion_{}.jpg'.format(i+1))
                elif self.test_dataset == 'MFFW2':
                    result_path = os.path.join(self.result_dir, 'MFFW2_fusion_{}.jpg'.format(i+1))
                elif self.test_dataset == 'grayscale_jpg':
                    result_path = os.path.join(self.result_dir, 'grayscale_fusion_{}.jpg'.format(i+1))

                save_image(self.denorm(fused_img.data.cpu()), result_path, nrow=1, padding=0)

                print('Saved focus map into {}...'.format(result_path))
                print('Used time is {}'.format(fusion_used_time))
                '''
                
                '''
                # ===================  testing save focus_map and refined focus map ======================
                A = A.to(self.device)
                B = B.to(self.device)

                focus_map = self.G.eval()(A, B)
                refined_focus_map = post_remove_small_objects(focus_map)
                # +++++++++++++++   save 520*520*3 uint8 black and white 0~255 black0 white255 rgb value same +++++++++++++ 
                focus_map = focus_map.data.cpu()
                focus_map = torch.squeeze(focus_map)
                focus_map = focus_map * 255
                focus_map = focus_map.numpy().astype('uint8')
                result3_path = os.path.join(self.focus_map_dir, 'focus_map{}.png'.format(i+1))
                skimage.io.imsave(result3_path, focus_map)
                print('Saved focus_map into {}...'.format(result3_path))
                           
                refined_focus_map = refined_focus_map.data.cpu()
                refined_focus_map = torch.squeeze(refined_focus_map)
                refined_focus_map = refined_focus_map * 255
                refined_focus_map = refined_focus_map.numpy().astype('uint8')
                result5_path = os.path.join(self.focus_map_pp_dir, 'focus_map_pp{}.png'.format(i+1))
                skimage.io.imsave(result5_path, refined_focus_map)
                print('Saved focus_map_pp into {}...'.format(result5_path))

                '''