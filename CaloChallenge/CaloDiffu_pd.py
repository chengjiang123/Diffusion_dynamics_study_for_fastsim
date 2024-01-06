import numpy as np
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import distributed as dist
from torch.autograd import Variable
from torchinfo import summary
from scipy import integrate
from utils import *
from models import *


class CaloDiffu_pd(nn.Module):
    """Diffusion based generative model"""
    def __init__(self, data_shape, config=None, R_Z_inputs = False, training_obj = 'noise_pred', nsteps = 400,factor = 2,
                    cold_diffu = False, E_bins = None, avg_showers = None, std_showers = None, NN_embed = None):
        super(CaloDiffu_pd, self).__init__()
        self._data_shape = data_shape
        self.nvoxels = np.prod(self._data_shape)
        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1
        self.factor = factor
        self.nsteps = 512//self.factor
        self.cold_diffu = cold_diffu
        self.E_bins = E_bins
        self.avg_showers = avg_showers
        self.std_showers = std_showers
        self.training_obj = training_obj
        self.shower_embed = self.config.get('SHOWER_EMBED', '')
        self.fully_connected = ('FCN' in self.shower_embed)
        self.NN_embed = NN_embed
        self.restart_info = config['restart_info']

        supported = ['noise_pred', 'mean_pred', 'hybrid']
        is_obj = [s in self.training_obj for s in supported]
        if(not any(is_obj)):
            print("Training objective %s not supported!" % self.training_obj)
            exit(1)


        if config is None:
            raise ValueError("Config file not given")
        
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank
        self.verbose = 1

        
        if(torch.cuda.is_available()): device = torch.device('cuda')
        else: device = torch.device('cpu')

        #Minimum and maximum maximum variance of noise
        self.beta_start = 0.0001
        self.beta_end = config.get("BETA_MAX", 0.02)

        #linear schedule
        schedd = config.get("NOISE_SCHED", "linear")
        self.discrete_time = True

        
        if("linear" in schedd): self.betas = torch.linspace(self.beta_start, self.beta_end, self.nsteps)
        elif("cosine" in schedd): 
            self.betas = cosine_beta_schedule(self.nsteps)
        elif("log" in schedd):
            self.discrete_time = False
            self.P_mean = -1.5
            self.P_std = 1.5
        else:
            print("Invalid NOISE_SCHEDD param %s" % schedd)
            exit(1)

        if(self.discrete_time):
            #precompute useful quantities for training
            self.alphas = 1. - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

            #shift all elements over by inserting unit value in first place
            alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

            self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

            self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
            
            
            self.sigma_min = self.sqrt_one_minus_alphas_cumprod[-1]

        self.time_embed = config.get("TIME_EMBED", 'sin')
        self.E_embed = config.get("COND_EMBED", 'sin')
        cond_dim = config['COND_SIZE_UNET']
        layer_sizes = config['LAYER_SIZE_UNET']
        block_attn = config.get("BLOCK_ATTN", False)
        mid_attn = config.get("MID_ATTN", False)
        compress_Z = config.get("COMPRESS_Z", False)


        if(self.fully_connected):
            #fully connected network architecture
            self.model = FCN(cond_dim = cond_dim, dim_in = config['SHAPE_ORIG'][1], num_layers = config['NUM_LAYERS_LINEAR'],
                    cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )

            self.R_Z_inputs = False

            summary_shape = [[1,config['SHAPE_ORIG'][1]], [1], [1]]


        else:
            RZ_shape = config['SHAPE_PAD'][1:]

            self.R_Z_inputs = config.get('R_Z_INPUT', False)
            self.phi_inputs = config.get('PHI_INPUT', False)

            in_channels = 1

            self.R_image, self.Z_image = create_R_Z_image(device, scaled = True, shape = RZ_shape)
            self.phi_image = create_phi_image(device, shape = RZ_shape)

            if(self.R_Z_inputs): in_channels = 3

            if(self.phi_inputs): in_channels += 1

            calo_summary_shape = list(copy.copy(RZ_shape))
            calo_summary_shape.insert(0, 1)
            calo_summary_shape[1] = in_channels

            calo_summary_shape[0] = 1
            summary_shape = [calo_summary_shape, [1], [1]]


            self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, block_attn = block_attn, mid_attn = mid_attn, 
                    cylindrical =  config.get('CYLINDRICAL', False), compress_Z = compress_Z, data_shape = calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )

        print("\n\n Model: \n")
        summary(self.model, summary_shape)

    #wrapper for backwards compatability
    def load_state_dict(self, d):
        if('noise_predictor' in list(d.keys())[0]):
            d_new = dict()
            for key in d.keys():
                key_new = key.replace('noise_predictor', 'model')
                d_new[key_new] = d[key]
        else: d_new = d

        return super().load_state_dict(d_new)

    def add_RZPhi(self, x):
        cats = [x]
        if(self.R_Z_inputs):

            batch_R_image = self.R_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)
            batch_Z_image = self.Z_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

            cats+= [batch_R_image, batch_Z_image]
        if(self.phi_inputs):
            batch_phi_image = self.phi_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

            cats += [batch_phi_image]

        if(len(cats) > 1):
            return torch.cat(cats, axis = 1)
        else: 
            return x
            
    
    def lookup_avg_std_shower(self, inputEs):
        idxs = torch.bucketize(inputEs, self.E_bins)  - 1 #NP indexes bins starting at 1 
        return self.avg_showers[idxs], self.std_showers[idxs]

    
    def noise_image(self, data = None, t = None, noise = None):

        if(noise is None): noise = torch.randn_like(data)

        if(t[0] <=0): return data

        if(self.discrete_time):
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
            out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
            return out
        else:
            print("NON DISCRETE TIME BAD")
            exit(1)
            
    def noise_image0(self, data = None, t = None, noise = None):

        
        if (data != None):
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
            out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
            return out
        else:
            print("NON DISCRETE TIME BAD")
            exit(1)
            
            
    def sqrt_alphas_cumprod0(self, t=None, data = None):
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        return extract(self.sqrt_alphas_cumprod, t, data.shape)
    
    
    
    def sqrt_one_minus_alphas_cumprod1(self, t=None, data = None):

        return extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
    
    def get_alpha_sigma(self, x, t):
        alpha = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha, sigma


    def compute_loss(self, data, energy, noise = None, t = None, loss_type = "l2", rnd_normal = None, energy_loss_scale = 1e-2, teacher = None):
        if noise is None:
            noise = torch.randn_like(data)
        
        


        if(self.discrete_time): 
            if(t is None): t = torch.randint(0, self.nsteps, (data.size()[0],), device=data.device).long()
            x_noisy = self.noise_image(data, t, noise=noise)
            sigma = None
            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)**2
        else:
            if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            x_noisy = data + torch.reshape(sigma, (data.shape[0], 1,1,1,1)) * noise
            sigma2 = sigma**2



        t_emb = self.do_time_embed(t, self.time_embed, sigma)


        pred = self.pred(x_noisy, energy, t_emb)

        weight = 1.
        x0_pred = None
        if('hybrid' in self.training_obj ):

            c_skip = torch.reshape(1. / (sigma2 + 1.), (data.shape[0], 1,1,1,1))
            c_out = torch.reshape(1./ (1. + 1./sigma2).sqrt(), (data.shape[0], 1,1,1,1))
            weight = torch.reshape(1. + (1./ sigma2), (data.shape[0], 1,1,1,1))

            #target = (data - c_skip * x_noisy)/c_out


            x0_pred = pred = c_skip * x_noisy + c_out * pred
            target = data

        elif('noise_pred' in self.training_obj):
            target = noise
            weight = 1.
            if('energy' in self.training_obj): 
                sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
                sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
                x0_pred = (x_noisy - sqrt_one_minus_alphas_cumprod_t * pred)/sqrt_alphas_cumprod_t
        elif('mean_pred' in self.training_obj):
            target = data
            weight = 1./ sigma2
            x0_pred = pred


        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(target, pred)
        elif loss_type == 'l2':
            if('weight' in self.training_obj):
                loss = (weight * ((pred - data) ** 2)).sum() / (torch.mean(weight) * self.nvoxels)
            else:
                loss = torch.nn.functional.mse_loss(target, pred)

        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(target, pred)
        elif loss_type == "pd":
            loss = teacher.compute_loss(data=data, E=E, noise = noise, t = t, loss_type = 'pd', energy_loss_scale = energy_loss_scale)
        else:
            raise NotImplementedError()

        if('energy' in self.training_obj):
            #sum total energy
            dims = [i for i in range(1,len(data.shape))]
            tot_energy_pred = torch.sum(x0_pred, dim = dims)
            tot_energy_data = torch.sum(data, dim = dims)
            loss_en = energy_loss_scale * torch.nn.functional.mse_loss(tot_energy_data, tot_energy_pred) / self.nvoxels
            loss += loss_en

        return loss
        
    def compute_loss_cm(self, x, E, t, ema_model, sample_algo = 'ddpm'):   

            x_noise = noise_image(x, t)

            #denoise 1 step using diffusion model
            x_prev = self.p_sample(x_noise, E, t, sample_algo = sample_algo)

            #predict ema model on denoised x
            x0_ema = ema_model.pred_cm(x_prev, E,t)
            #predict model on x
            x0 = model.pred_cm(x_noise, E,t)

            loss = torch.nn.functional.mse_loss(x0_ema, x0)
            return loss

    def do_time_embed(self, t = None, embed_type = "identity",  sigma = None,):
        
        if(sigma is None): sigma = self.sqrt_one_minus_alphas_cumprod.to(t.device)[t]
                

        if(embed_type == "identity" or embed_type == 'sin'):
            return t
        if(embed_type == "scaled"):
            return t/self.nsteps
        if(embed_type == "sigma"):
                #return torch.sqrt(self.betas[t]).to(t.device)
            return sigma.to(t.device)
        if(embed_type == "log"):
            return 0.5 * torch.log(sigma).to(t.device)
        #else:
            #if(embed_type == "log"):
            #return 0.5 * torch.log(sigma).to(t.device)
        #else:
            #return sigma
    def get_steps(self, x, num_step, min_t, max_t, rho):

        step_indices = torch.arange(num_step, dtype=torch.float32, device=x.device)
        t_steps = (max_t ** (1 / rho) + step_indices / (num_step - 1) * (min_t ** (1 / rho) - max_t ** (1 / rho))) ** rho
        return t_steps
    
    def linear_multistep_coeff(self, order, t, i, j):
        if order - 1 > i:
            raise ValueError(f'Order {order} too high for step {i}')
        def fn(tau):
            prod = 1.
            for k in range(order):
                if j == k:
                    continue
                prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
            return prod
        return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


    def edm_sampler(
        self, x, E, sample_algo = 'euler', randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.02, sigma_max=100, rho=7,order=4,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1.,eta=1.,beta_d=19.9, beta_min=0.1, eps_s=1e-3,
        restart_info='{"0": [4, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 5, 0.59, 1.09], "3": [4, 5, 0.30, 0.59], "4": [6, 6, 0.06, 0.30]}', restart_gamma=0
    ):

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None
        h_last = None
        #N = net.img_channels * net.img_resolution * net.img_resolution
        # Adjust noise levels based on what's supported by the network.
        #sigma_min = max(sigma_min, net.sigma_min)
        #sigma_max = min(sigma_max, net.sigma_max)

        gen_size = x.shape[0]

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x.device)
        
        #default edm karras
        #t_steps = self.get_steps(num_step = num_steps, x=x, min_t=sigma_min, max_t=sigma_max, rho=rho)
        #ramp = torch.linspace(0,1,num_steps)
        #t_steps = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        #t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    #sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            
        #lu
        #rho=1
        #lambda_min=np.log(sigma_min)
        #lambda_max=np.log(sigma_max)
        #t_steps = self.get_steps(num_step = num_steps, x=x, min_t=lambda_min, max_t=lambda_max, rho=rho)
        #sigmas = (lambda_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    #lambda_min ** (1 / rho) - lambda_max ** (1 / rho))) ** rho
        #t_steps = torch.exp(sigmas)
        
        #vp-sde
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / eps_s - np.log(sigma_max ** 2 + 1)) / (eps_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d                
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (eps_s - 1)
        t_steps = torch.sqrt(torch.exp(vp_beta_d * orig_t_steps ** 2 / 2 + vp_beta_min * orig_t_steps) - 1)
        
        #ddpm
        #betas = cosine_beta_schedule(num_steps)
        #alphas = 1 - betas
        #alphas_cumprod = torch.cumprod(self.alphas, axis = 0)
        #sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
        #t_steps = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        
        #time_steps = list(range(0, num_steps, 1))
        #time_steps.reverse()
        
        
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        #print(t_steps)
        total_step = len(t_steps)

        x_next = x.to(torch.float32) * t_steps[0]

        # Main sampling loop.
        if (sample_algo == 'dpmpp_2m_vp'):
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_hat = x_next
                t_hat = torch.as_tensor(t_cur)
                t_hat_full = torch.full((gen_size,), t_hat, device=x.device)
                #times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
                denoised = self.denoise(x_hat, E, t_hat_full).to(torch.float32)
                t1, t2 = t_fn(t_hat), t_fn(t_next)
                h = t2 - t1
                if (sample_algo == 'dpmpp_2m_vp')  and (old_denoised == None):
                    x_next = (sigma_fn(t2) / sigma_fn(t1)) * x_hat - (-h).expm1() * denoised
                elif (i < num_steps - 1):
                    r = h_last / h
                    denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                    x_next = (sigma_fn(t2) / sigma_fn(t1)) * x_hat - (-h).expm1() * denoised_d
                old_denoised = denoised
                h_last = h
        if (sample_algo == 'dpmpp_2m_sde'):
            #print('only use')
            for i in range(len(t_steps) - 1):
                x_hat = x_next
                t_hat = torch.as_tensor(t_steps[i])
                t_hat_full = torch.full((gen_size,), t_hat, device=x.device)
                denoised = self.denoise(x_hat, E, t_hat_full).to(torch.float32)
                if t_steps[i + 1] == 0:
                    x_next = denoised
                else:
                    # DPM-Solver++(2M) SDE
                    t, s = -t_steps[i].log(), -t_steps[i + 1].log()
                    h = s - t
                    eta_h = eta * h
                    
                    x_next = t_steps[i + 1] / t_steps[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

                    if old_denoised is not None:
                        r = h_last / h
                        x_next = x_hat + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                        #print(old_denoised)
                        #print(x_next)

                old_denoised = denoised
                h_last = h

                
        if (sample_algo == 'lms'):
            #t_steps_cpu = t_steps.detach().cpu().numpy()
            t_steps_cpu = t_steps
            ds = []
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_hat = x_next
                t_hat = torch.as_tensor(t_cur)
                t_hat_full = torch.full((gen_size,), t_hat, device=x.device)
                denoised = self.denoise(x_hat, E, t_hat_full).to(torch.float32)
                d_cur = (x_hat - denoised) / t_hat
                ds.append(d_cur)
                if len(ds) > order:
                    ds.pop(0)
                cur_order = min(i + 1, order)
                coeffs = [self.linear_multistep_coeff(cur_order, t_steps_cpu, i, j) for j in range(cur_order)]
                x_next = x_hat + sum(coeff * d_cur for coeff, d_cur in zip(coeffs, reversed(ds)))

        else:
            #print('test')
            # {[num_steps, number of restart iteration (K), t_min, t_max], ... }
            import json
            restart_list = json.loads(restart_info) if restart_info != '' else {}
            # cast t_min to the index of nearest value in t_steps
            restart_list = {int(torch.argmin(abs(t_steps - v[2]), dim=0)): v for k, v in restart_list.items()}
            # dist.print0(f"restart configuration: {restart_list}")


            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N_main -1
                x_cur = x_next
                # Increase noise temporarily.
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                t_hat = torch.as_tensor(t_cur + gamma * t_cur)
                #print(t_hat)
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
                # Euler step.
                t_hat_full = torch.full((gen_size,), t_hat, device=x.device)


                #pred = self.pred(x_hat, E, t_hat_full)


                denoised = self.denoise(x_hat, E, t_hat_full).to(torch.float32)
                #print(denoised)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply edm 2nd order correction.
                if (sample_algo == 'edm')  and (i < num_steps - 1):
                    #print('use edm')
                    t_next_full = torch.full((gen_size,), t_next, device=x.device)
                    denoised = self.denoise(x_next, E, t_next_full).to(torch.float32)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                # custom dpm2/edm 2nd order correction.    
                if (sample_algo == 'dpm2')  and (i < num_steps - 1):
                    #print('use dpm2')
                    t_mid = t_hat.log().lerp(t_next.log(), 0.5).exp()
                    dt_1 = t_mid - t_hat
                    dt_2 = t_next - t_hat
                    x_2 = x_hat + d_cur * dt_1
                    t_mid_full = torch.full((gen_size,), t_mid, device=x.device)
                    denoised_2 = self.denoise(x_2, E, t_mid_full).to(torch.float32)
                    d_2 = (x_2 - denoised_2) / t_mid
                    x_next = x_hat + d_2*dt_2


                if (sample_algo == 'restart'):

                    # ================= restart ================== #
                    if i + 1 in restart_list.keys():
                        restart_idx = i + 1

                        for restart_iter in range(restart_list[restart_idx][1]):

                            new_t_steps = self.get_steps(min_t=t_steps[restart_idx], max_t=restart_list[restart_idx][3], num_step=restart_list[restart_idx][0], rho=rho, x=x)
                            print(f"restart at {restart_idx} with {new_t_steps}")
                            new_total_step = len(new_t_steps)

                            x_next = x_next + randn_like(x_next) * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt() * S_noise


                            for j, (t_cur, t_next) in enumerate(zip(new_t_steps[:-1], new_t_steps[1:])):  # 0, ..., N_restart -1

                                x_cur = x_next
                                gamma = restart_gamma if S_min <= t_cur <= S_max else 0
                                t_hat = torch.as_tensor(t_cur + gamma * t_cur)

                                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)


                                t_hat_full = torch.full((gen_size,), t_hat, device=x.device)

                                #pred = self.pred(x_hat, E, t_hat_full)


                                denoised = self.denoise(x_hat, E, t_hat_full).to(torch.float32)
                                d_cur = (x_hat - denoised) / (t_hat)
                                x_next = x_hat + (t_next - t_hat) * d_cur

                                # Apply 2nd order correction.
                                if (sample_algo == 'restart') and (j < new_total_step - 2 or new_t_steps[-1] != 0):
                                    t_next_full = torch.full((gen_size,), t_next, device=x.device)
                                    denoised = self.denoise(x_next, E, t_next_full).to(torch.float32)
                                    d_prime = (x_next - denoised) / t_next
                                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


    def edm_sampler0( self, x, E, sample_algo = 'euler', randn_like=torch.randn_like, num_steps=400, sigma_min=0.002, sigma_max=10, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,):
        # Adjust noise levels based on what's supported by the network.

        #sigma_min = max(sigma_min, net.sigma_min)
        #sigma_max = min(sigma_max, net.sigma_max)

        gen_size = x.shape[0]
        #print(gen_size)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        
        #print(t_steps)

        # Main sampling loop.
        x_next = x.to(torch.float32) * t_steps[0]
        print(x_next.shape)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            #print(gamma)
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            #print(t_hat)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.

            t_hat_full = torch.full((gen_size,), t_hat, device=x.device)

            pred = self.pred(x_hat, E, t_hat_full)
            #print(pred)
            denoised = self.denoise(x_hat, E, t_hat_full).to(torch.float32)
            #print(denoised)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur



            # Apply 2nd order correction.
            if (sample_algo == 'edm') and (i < num_steps - 1):
                t_next_full = torch.full((gen_size,), t_next, device=x.device)
                denoised = self.denoise(x_next, E, t_next_full).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


    def pred(self, x, E, t_emb):

        #if(self.NN_embed is not None): x = self.NN_embed.enc(x).to(x.device)
        out = self.model(self.add_RZPhi(x), E, t_emb)
        #if(self.NN_embed is not None): out = self.NN_embed.dec(out).to(x.device)
        return out
    
    def pred_cm(self, x, E, t_emb):

        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t_emb, x.shape)
        #print(x.shape)

        out = self.model(self.add_RZPhi(x), E, t_emb)


        sigma2 = (sigma - self.sigma_min)**2
        c_skip = torch.reshape(1. / (sigma2 + 1.), (x.shape[0], 1,1,1,1))
        c_out = torch.reshape(1./ (1. + 1./sigma2).sqrt(), (x.shape[0], 1,1,1,1))

        return c_skip * x + c_out * out

    def denoise(self, x, E,t_emb):
        pred = self.pred(x, E, t_emb)
        if('mean_pred' in self.training_obj):
            return pred
        elif('hybrid' in self.training_obj):
            sigma2 = (t_emb**2).reshape(-1,1,1,1,1)
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()
            #c_out = -1.*torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            return c_skip * x + c_out * pred


    @torch.no_grad()
    def p_sample(self, x, E, t, cold_noise_scale = 0., noise = None, sample_algo = 'ddpm', debug = False):
        #reverse the diffusion process (one step)



        if(noise is None): 
            noise = torch.randn(x.shape, device = x.device)
            if(self.cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
                noise = self.gen_cold_image(E, cold_noise_scale, noise)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        #print(t, self.sqrt_one_minus_alphas_cumprod[t])
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)

        t_emb = self.do_time_embed(t, self.time_embed)


        pred = self.pred(x, E, t_emb)
        if('noise_pred' in self.training_obj):
            noise_pred = pred
            x0_pred = None
        elif('mean_pred' in self.training_obj):
            x0_pred = pred
            noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t
        elif('hybrid' in self.training_obj):

            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)**2
            #sigma2 = extract(self.betas, t, x.shape)**2
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()
            #c_out = -1.*torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            x0_pred = c_skip * x + c_out * pred
            noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t

        


        if(sample_algo == 'ddpm'):
            # Sampling algo from https://arxiv.org/abs/2006.11239
            # Use results from our model (noise predictor) to predict the mean of posterior distribution of prev step
            post_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            out = post_mean + torch.sqrt(posterior_variance_t) * noise 
            if t[0] == 0: out = post_mean

        elif(sample_algo == 'ddim'):
            if(x0_pred is None): x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
            if t[0] == 0: out = x0_pred
            else:
                t_next = t-1
                sqrt_alphas_cumprod_t_next = extract(self.sqrt_alphas_cumprod, t_next, x.shape)

                c1 =  torch.sqrt(1. - sqrt_alphas_cumprod_t_next **2 - posterior_variance_t **2)
                out = sqrt_alphas_cumprod_t_next * x0_pred + c1 * noise_pred + sqrt_alphas_cumprod_t * noise
                #print(torch.mean(out), torch.mean(x0_pred), torch.mean(c1 * noise_pred), torch.mean(sqrt_alphas_cumprod_t * noise))

        elif(sample_algo == 'cold_step'):
            post_mean = x - noise_pred * sqrt_one_minus_alphas_cumprod_t
            post_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            out = post_mean
            #out = post_mean + torch.sqrt(posterior_variance_t) * noise 


        elif('cold' in sample_algo):

            if(x0_pred is None):
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t

            #algo 2 from cold diffu paper
            # x_t-1 = x(t, eps_t) - D(x0, t, eps_t) + D(x0, t-1, eps_t-1)
            #Must use same eps for x_t and D(x0, t), otherwise unstable
            if('cold2' in sample_algo):
                out = x - self.noise_image(x0_pred, t, noise = self.prev_noise) + self.noise_image(x0_pred, t-1, noise = noise)
                self.prev_noise = noise
            else:
            #algo 1
                out = self.noise_image(x0_pred, t-1, noise = noise)
            #print(torch.mean(out), torch.std(out))



        if(debug): 
            if(x0_pred is None):
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
            return out, x0_pred
        return out
    
   
    
    
    def gen_cold_image(self, E, cold_noise_scale, noise = None):

        avg_shower, std_shower = self.lookup_avg_std_shower(E)

        if(noise is None):
            noise = torch.randn_like(avg_shower, dtype = torch.float32)

        cold_scales = cold_noise_scale

        return torch.add(avg_shower, cold_scales * (noise * std_shower))




    @torch.no_grad()
    def Sample(self, E, num_steps = 200, cold_noise_scale = 0., sample_algo = 'ddpm', debug = False, sample_offset = 0, sample_step = 1):
        """Generate samples from diffusion model.
        
        Args:
        E: Energies
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        
        Returns: 
        Samples.
        """

        print("SAMPLE ALGO : %s" % sample_algo)

        # Full sample (all steps)
        device = next(self.parameters()).device


        gen_size = E.shape[0]
        # start from pure noise (for each example in the batch)
        gen_shape = list(copy.copy(self._data_shape))
        gen_shape.insert(0,gen_size)

        #start from pure noise
        x_start = torch.randn(gen_shape, device=device)

        avg_shower = std_shower = None
        if(self.cold_diffu): #cold diffu starts using avg images
            x_start = self.gen_cold_image(E, cold_noise_scale)


        start = time.time()

        if(sample_algo == 'euler' or sample_algo == 'lms' or sample_algo == 'edm' or sample_algo == 'restart' or sample_algo == 'dpm2' or sample_algo == 'dpmpp_2m_vp' or sample_algo == 'dpmpp_2m_sde'):
            S_churn = 30
            S_min = 0.01
            S_max = 1
            S_noise = 1.003
            sigma_min = 0.002
            sigma_max = 80.0
            eta = 1.0
            rho = 7.0
            order = 4

            x = self.edm_sampler(x_start,E, num_steps = num_steps, sample_algo = sample_algo, sigma_min = sigma_min, sigma_max = sigma_max, 
                    S_churn = S_churn, S_min = S_min, S_max = S_max, S_noise = S_noise, eta=eta, rho=rho, order = order, restart_info=self.restart_info)

        else:
            x = x_start
            fixed_noise = None
            if('fixed' in sample_algo): 
                print("Fixing noise to constant for sampling!")
                fixed_noise = x_start
            xs = []
            x0s = []
            self.prev_noise = x_start

            time_steps = list(range(0, num_steps - sample_offset, sample_step))
            time_steps.reverse()

            for time_step in time_steps:      
                times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
                out = self.p_sample(x, E, times, noise = fixed_noise, cold_noise_scale = cold_noise_scale, sample_algo = sample_algo, debug = debug)
                if(debug): 
                    x, x0_pred = out
                    xs.append(x.detach().cpu().numpy())
                    x0s.append(x0_pred.detach().cpu().numpy())
                else: x = out

        end = time.time()
        times_array = end - start
        print("Time for sampling {} events is {} seconds".format(gen_size,end - start), flush=True)
        if(debug):
            return x.detach().cpu().numpy(), xs, x0s, times_array
        else:   
            return x.detach().cpu().numpy(), times_array

    
        
