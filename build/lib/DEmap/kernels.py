import torch
import gpytorch

#--- Define custom Geodesic Kernel ---
class GeodesicMaternKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    

    def __init__(self, nu=1.5, length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, 2.5")
        self.nu = nu

   
    def forward(self, x1, x2, diag=False, **params):
        # center & scale
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
        x1_ = (x1 - mean).div(self.lengthscale)

        x2_ = (x2 - mean).div(self.lengthscale)

        # normalize
        x1_unit = x1_ / x1_.norm(dim=-1, keepdim=True)
        x2_unit = x2_ / x2_.norm(dim=-1, keepdim=True)

        # geodesic distance
        cos_theta = torch.matmul(x1_unit, x2_unit.transpose(-2, -1))
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        distance = torch.acos(cos_theta)

        # Matern covariance
        if self.nu == 0.5:
            constant = 1.0
        elif self.nu == 1.5:
            constant = 1.0 + math.sqrt(3) * distance
        elif self.nu == 2.5:
            constant = 1.0 + math.sqrt(5) * distance + 5.0/3.0 * distance**2

        covar = constant * torch.exp(-math.sqrt(2*self.nu) * distance)

        if diag:
            covar = covar.diagonal(dim1=-2, dim2=-1)

        return covar


# Exact GP object using Geodesic matern
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = GeodesicMaternKernel(lengthscale=0.1, nu=0.5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


