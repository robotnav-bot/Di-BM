from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
import numpy as np
import matplotlib.pyplot as plt

class DiffusionUnetImagePolicy(BaseImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,

        num_experts=5,
        num_samples_per_expert=4,
        gamma=10,
        beta=0.1,

        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps


        
        self.num_experts = num_experts
        self.num_samples_per_expert = num_samples_per_expert
        self.gamma = gamma
        self.beta = beta

        self.gating_network =nn.Sequential(
            nn.Linear(global_cond_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,num_experts),
        )
        self._log_weights = -torch.log(torch.ones(self.num_experts) * self.num_experts)[None, :]



        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            num_experts=num_experts
        )    
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.pco_list = []
    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        use_expert_i=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond, use_expert_i=use_expert_i)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        self.Z=torch.Tensor([[18.739338, 23.333763, 13.635132,  6.825393, 76.133156]]).to(global_cond.device)

        gating_logits = self.gating_network(global_cond)
        # print(torch.exp(gating_logits))
        poe = torch.exp(gating_logits)/self.Z



        peo = poe/poe.sum()
        use_expert_i = peo.argmax(dim=1)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            use_expert_i=use_expert_i,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result, peo

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        # print("batch action:", batch['action'][0, 0, -4:])
        nactions = self.normalizer['action'].normalize(batch['action'])
        # print("nactions:", nactions[0, 0, -4:])
        assert self.obs_as_global_cond


        # global_cond = self.obs_encoder(nobs)
        if self.obs_as_global_cond:
            batch_size = nactions.shape[0]

            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        # else:
        #     # reshape B, T, ... to B*T
        #     this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # reshape back to B, T, Do
        #     nobs_features = nobs_features.reshape(batch_size, horizon, -1)
        #     cond_data = torch.cat([nactions, nobs_features], dim=-1)
        #     trajectory = cond_data.detach()


        gating_logits = self.gating_network(global_cond)

        log_gating_probs = torch.log_softmax(gating_logits, dim=0) #(B,num_experts)


        # get prob dist for each expert
        expert_gating = torch.exp(log_gating_probs) 
        sampled_indices = self.sample_indices_per_expert(expert_gating) #(num_experts, num_samples_per_expert)
        

        global_cond_expert = torch.cat([global_cond[idx] for idx in sampled_indices], dim=0) #(num_experts*num_samples_per_expert, ...)
        nactions_expert = torch.cat([nactions[idx] for idx in sampled_indices], dim=0) #(num_experts*num_samples_per_expert, ...)
        
        assert global_cond_expert.shape[0] == self.num_experts * self.num_samples_per_expert
        assert nactions_expert.shape[0] == self.num_experts * self.num_samples_per_expert

        use_expert_i = torch.arange(self.num_experts).repeat_interleave(self.num_samples_per_expert).to(log_gating_probs.device)  #(num_experts*num_samples_per_expert, )

        expert_loss=self.expert_forward_together(global_cond_expert, nactions_expert, use_expert_i)

        # update gating network p(o|e)
        with torch.no_grad():
            log_resps=self.log_resps(log_gating_probs).detach() # log(p(e|o)) (B,num_experts)

        gating_loss=[]
        KL_loss=[]
        for expert_idx in range(self.num_experts):
            with torch.no_grad():
                loss=self.gating_forward(nactions=nactions, global_cond=global_cond, use_expert_i=expert_idx) #(B,)
            
            entropy = -log_gating_probs[:,expert_idx] #(B,)

            gating_loss.append((torch.exp(log_gating_probs[:,expert_idx])*(loss - self.beta * log_resps[:,expert_idx] - self.beta * entropy)).mean())
            KL_loss.append((torch.exp(log_gating_probs[:,expert_idx])*( -log_resps[:,expert_idx] - entropy)).mean())

        gating_loss=torch.stack(gating_loss).mean()
        KL_loss=torch.stack(KL_loss).mean()

        loss = expert_loss + self.gamma * gating_loss

        return loss, gating_loss, expert_loss, KL_loss



    def compute_datasets_Z(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        # print("batch action:", batch['action'][0, 0, -4:])
        nactions = self.normalizer['action'].normalize(batch['action'])
        # print("nactions:", nactions[0, 0, -4:])
        assert self.obs_as_global_cond


        # global_cond = self.obs_encoder(nobs)
        if self.obs_as_global_cond:
            batch_size = nactions.shape[0]

            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
            


        gating_logits = self.gating_network(global_cond)

        gating_logits = torch.exp(gating_logits)

        Z = torch.sum(gating_logits, dim=0) #(num_experts,)
        
        return Z




    def gating_forward(self, nactions, global_cond, use_expert_i):
        
        trajectory = nactions
        cond_data = trajectory

        use_expert_i=(torch.ones(nactions.shape[0])*use_expert_i).to(torch.long).to(nactions.device)


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, local_cond=None, global_cond=global_cond, use_expert_i=use_expert_i)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)


        loss = reduce(loss, 'b ... -> b', 'mean')

        return loss #(B,)

    def expert_forward_together(self, global_cond, nactions, use_expert_i):

        trajectory = nactions
        cond_data = trajectory

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, local_cond=None, global_cond=global_cond, use_expert_i=use_expert_i)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss


    # p(o)=sum(p(o|e)p(e))
    def log_cmp_m_ctxt_densities(self, log_gating_probs):
        # gating_probs: p(o|e)

        exp_arg = log_gating_probs + self._log_weights.to(log_gating_probs.device) #log(p(o|e)p(e))
        log_marg_ctxt_densities = torch.logsumexp(exp_arg, dim=1)
        return log_marg_ctxt_densities

    # p(e|o)=p(o|e)p(e)/p(o) 
    def log_resps(self, log_gating_probs):
        # gating_probs: p(o|e)

        log_marg_ctxt_densities = self.log_cmp_m_ctxt_densities(log_gating_probs) # log(p(o))

        log_gating_probs = log_gating_probs + self._log_weights.to(log_gating_probs.device) - log_marg_ctxt_densities[:, None]
        return log_gating_probs


    def sample_indices_per_expert(self, expert_gating: torch.Tensor):
        B, num_experts = expert_gating.shape
        indices_per_expert = []

        for e in range(num_experts):
            probs = expert_gating[:, e]
            sampled_indices = torch.multinomial(probs, self.num_samples_per_expert, replacement=False)
            indices_per_expert.append(sampled_indices)
        indices_per_expert=torch.stack(indices_per_expert,dim=0)
        return indices_per_expert #(num_experts, num_samples_per_expert)

