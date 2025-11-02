import torch
import gpytorch
import math
# Define custom Geodesic Kernel 
class GeodesicMaternKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, lengthscale=1.0, nu=0.5, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(torch.tensor(lengthscale)))
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    def forward(self, x1, x2, diag=False, **params):
        # normalize inputs to lie on unit sphere
        x1_norm = x1 / x1.norm(dim=-1, keepdim=True)
        x2_norm = x2 / x2.norm(dim=-1, keepdim=True)

        # geodesic distance
        cos_theta = torch.matmul(x1_norm, x2_norm.transpose(-2, -1)).clamp(-1.0, 1.0)
        dist = torch.acos(cos_theta)

        # Matern kernel (nu=3/2)
        if self.nu == 1.5:
            sqrt3_d = torch.sqrt(torch.tensor(3.0)) * dist / self.lengthscale
            covar = (1.0 + sqrt3_d) * torch.exp(-sqrt3_d)
        elif self.nu == 2.5:
            sqrt5_d = torch.sqrt(torch.tensor(5.0)) * dist / self.lengthscale
            covar = (1.0 + sqrt5_d + 5.0*dist**2/(3.0*self.lengthscale**2)) * torch.exp(-sqrt5_d)
        elif self.nu == 0.5:
            covar = torch.exp(-dist / self.lengthscale)
        else:
            # general case using Bessel function (less common)
            raise NotImplementedError("Only nu=0.5, 1.5, 2.5 are implemented in closed form.")

        if diag:
            return covar.diag()
        else:
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
