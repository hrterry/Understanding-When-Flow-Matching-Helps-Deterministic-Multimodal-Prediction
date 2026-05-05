import torch
from .noise import PriorSampler


class Interpolant:
    def __init__(self, prior_sample_type, normalize=True, device=None, **kwargs):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # t translated，translated（translated）
        self.use_t_bounds = kwargs.get('use_t_bounds', True)
        if self.use_t_bounds:
            self.t_min = float(kwargs.get('t_min', 1e-3))
            self.t_max = float(kwargs.get('t_max', 0.999))
            if self.t_min < 0.0:
                self.t_min = 0.0
            if self.t_max > 1.0:
                self.t_max = 1.0
            if self.t_min > self.t_max:
                self.t_min, self.t_max = self.t_max, self.t_min

        # translated（corrupt_exp translated vanilla translated α、α'）
        self.alpha_schedule = kwargs.get('alpha_schedule', 'linear')  # linear, quad, cos, sigm

        # Rectified-Flow translated：translated t，translated t' = r t / (1 + (r-1)t)，r translated
        self.t_schedule = kwargs.get('t_schedule', 'linear')  # linear | logit_normal
        self.logit_normal_mu = float(kwargs.get('logit_normal_mu', 0.0))
        self.logit_normal_sigma = float(kwargs.get('logit_normal_sigma', 1.0))
        self.r = float(kwargs.get('r', 1.0))
        if self.r <= 0:
            self.r = 1.0

        self.prior_sampler = PriorSampler(prior_sample_type, device=self.device, **kwargs)
        self.normalize = normalize
    
    def alpha(self, t, kind=None):
        """
        translatedalphatranslated
        
        Args:
            t: translated [B] translated
            kind: translated，translatedNonetranslatedself.alpha_schedule
        
        Returns:
            alphatranslated，translatedttranslated
        """
        if kind is None:
            kind = self.alpha_schedule
            
        if kind == "linear":
            return t
        elif kind == "quad":
            return t ** 2
        elif kind == "cos":
            # translated
            return 0.5 - 0.5 * torch.cos(torch.pi * t)
        elif kind == "sigm":
            # Stranslated，translated
            return torch.sigmoid(4 * (t - 0.5))
        else:
            return t
    
    def alpha_derivative(self, t, kind=None):
        """
        dα/dt，translated vanilla translated v* = α'(t)(x1 - x0)。

        Args:
            t: translated [B] translated
            kind: translated，translatedNonetranslatedself.alpha_schedule
        
        Returns:
            alphatranslated，translatedttranslated
        """
        if kind is None:
            kind = self.alpha_schedule
            
        if kind == "linear":
            return torch.ones_like(t)
        elif kind == "quad":
            return 2 * t
        elif kind == "cos":
            return 0.5 * torch.pi * torch.sin(torch.pi * t)
        elif kind == "sigm":
            # sigmoidtranslated sigmoid(x) * (1 - sigmoid(x)) * 4
            sigm_val = torch.sigmoid(4 * (t - 0.5))
            return 4 * sigm_val * (1 - sigm_val)
        else:
            return torch.ones_like(t)
    
    def sample_from_prior(self, shape):
        exp = self.prior_sampler.sample(shape).to(self.device)
        if self.normalize:
            exp = torch.log(exp + 1)
        return exp

    def _apply_resolution_shift(self, t):
        """t' = r t / (1 + (r-1) t)，r translated self.r，translated [0,1] translated [0,1]。"""
        r = self.r
        if abs(r - 1.0) < 1e-12:
            return t
        denom = 1.0 + (r - 1.0) * t
        return r * t / (denom + 1e-8)

    def get_timestep(self, batch_size, device=None):
        """
        translated：LINEAR translated batch translated torch.rand([0,1])；LOGIT_NORMAL translated u~N(μ,σ) translated t=sigmoid(u)。
        translated t translated resolution-based shifting。
        """
        if device is None:
            device = self.device
        if self.t_schedule == 'logit_normal':
            u = torch.randn(batch_size, device=device, dtype=torch.float32) * self.logit_normal_sigma + self.logit_normal_mu
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device, dtype=torch.float32)
            if self.use_t_bounds:
                t = self.t_min + (self.t_max - self.t_min) * t
        t = self._apply_resolution_shift(t)
        if self.use_t_bounds:
            t = torch.clamp(t, self.t_min, self.t_max)
        return t

    def get_sampler_timesteps(self, num_steps, device=None):
        """
        translated：translated resolution shift。translated t = s/num_steps，s=0..num_steps-1（translated test translated t=s/S translated）。
        """
        if device is None:
            device = self.device
        s = torch.arange(num_steps, device=device, dtype=torch.float32)
        t = s / float(num_steps)
        t = self._apply_resolution_shift(t)
        if self.use_t_bounds:
            t = torch.clamp(t, self.t_min, self.t_max)
        return t

    def sample_t(self, shape):
        """translated：translated get_timestep(shape[0])。"""
        batch_size = shape[0] if isinstance(shape, (tuple, list)) else int(shape)
        return self.get_timestep(batch_size)

    def corrupt_exp(self, exp):
        # exp: [B, n_cells, n_genes]
        t = self.get_timestep(exp.shape[0], device=self.device)
        exp_0 = self.sample_from_prior(exp.shape).to(self.device)
        
        # translatedalphatranslated
        alpha_t = self.alpha(t)
        exp_t = exp_0 * (1 - alpha_t[:, None, None]) + exp * alpha_t[:, None, None]
        return exp_t, exp_0, t

    def sample_from_prior_latent(self, shape):
        """
        Latent translated：translated N(0,1)（torch.randn）。
        translated prior_sampler（zinb/gaussian translated）translated，translated log1p。
        shape: (B, M, latent_dim)
        """
        return torch.randn(*shape, device=self.device, dtype=torch.float32)

    def corrupt_latent(self, z1):
        """
        translated latent translated corrupt_exp translated α translated：z_t = (1-α)z0 + α z1，translated z0 ~ N(0,I)。
        z1: [B, M, latent_dim]
        returns: z_t, z0, t
        """
        t = self.get_timestep(z1.shape[0], device=self.device)
        z0 = self.sample_from_prior_latent(z1.shape).to(self.device)
        alpha_t = self.alpha(t)
        z_t = z0 * (1 - alpha_t[:, None, None]) + z1 * alpha_t[:, None, None]
        return z_t, z0, t

    def denoise(self, v_pred, exp_t, t, d_t):
        # v_pred: model-predicted velocity field (Y - Y0), same shape as exp_t
        # exp_t: [B, n_cells, n_genes]
        # t: [B]
        # d_t: [B]
        return exp_t + d_t[:, None, None] * v_pred
    
    def get_pathflow_target_velocity(self, exp, exp_0, t):
        """
        PathFlow target velocity:
        v* = (x1 - x0) / (1 - t)
        """
        return (exp - exp_0) / (1 - t[:, None, None] + 1e-8)
        # return (exp - exp_0) 

    def get_vanilla_flow_target(self, exp, exp_0, t):
        """
        vanilla translated（translated corrupt_exp translated x_t = (1-α)x_0 + α x_1 translated）：
        v* = α'(t) (x1 - x0) = d x_t / dt
        """
        alpha_deriv = self.alpha_derivative(t)
        return (exp - exp_0) * alpha_deriv[:, None, None]

    def get_control_target_u_star(self, x1, x0, t, delta, eps=1e-8):
        """
        flow_loss_type=control translated target velocity，translated（translated corrupt translated α translated）::

            u*(t) = exp(-(1-t)/Δ) / ( (Δ/2)(1 - exp(-2/Δ)) ) * ( x1 - exp(-1/Δ) x0 )

        Args:
            x1, x0: [B, M, C]
            t: [B]
            delta: Δ，translated
        """
        device, dtype = x1.device, x1.dtype
        d = torch.as_tensor(delta, device=device, dtype=dtype)
        t_f = t.to(device=device, dtype=dtype)
        numer = torch.exp(-(1.0 - t_f) / d)
        denom = (d / 2.0) * (1.0 - torch.exp(-2.0 / d))
        displacement = x1 - torch.exp(-1.0 / d) * x0
        scale = (numer / (denom + eps)).view(-1, 1, 1)
        return scale * displacement
