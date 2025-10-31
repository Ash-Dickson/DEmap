import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")   # headless, no GUI
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, List, Literal
from DEmap.sphere_utils import sphere_to_polar

class Plot_Tools:
    

    def __init__(self, input_points, output_values, theta_range, phi_range):
        
        self.input_points = input_points
        self.output_values = output_values
        self.theta_range = theta_range
        self.phi_range = phi_range
   

    
    def plot(self, plot_type: Literal["var", "mean"], fname=None):

        """
        Plot a heatmap of the TDE surface in polar coordinates.

        Parameters:
            plot_type (Literal["var", "mean"]): Type of plot to generate. If "mean", plot the mean TDE. If "var", plot the variance of the TDE.
            fname (Optional[str]): If not None, name of the output file to save the plot to.

        Returns:
            None
        """
        theta, phi = sphere_to_polar(self.input_points)

         # Enforce periodicity so plot works correctly
        phi_extended = np.concatenate([phi, phi - 2*np.pi, phi + 2*np.pi])
        # Repeat theta and output to match size of periodic phi
        theta_extended = np.tile(theta, 3)
        values_extended = np.tile(self.output_values, 3)

        # phi_grid, theta_grid, grid_values, phi, theta = self.generate_sphere_data(input=input_points, output=output_values)
        n_phi, n_theta = 300, 150
        phi_grid = np.linspace(self.phi_range[0], self.phi_range[1], n_phi)
        theta_grid = np.linspace(self.theta_range[0], self.theta_range[1], n_theta)
        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)

        grid_values = griddata(
            (phi_extended, theta_extended),
            values_extended,
            (phi_mesh, theta_mesh),
            method='linear'
        )

        # Plot on polar projection
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
       

        # Apply limits (convert θ to degrees since you plot np.degrees(theta))
        ax.set_rlim(np.degrees(self.theta_range[0]), np.degrees(self.theta_range[1]))
        ax.set_thetalim(self.phi_range[0], self.phi_range[1])
        
        
        ax.set_theta_zero_location("E")
        
        c = ax.pcolormesh(phi_grid, np.degrees(theta_grid), grid_values,
                        cmap='plasma' if plot_type == "mean" else 'viridis', shading='gouraud')
        # Add contour lines on top
        contour_levels = np.linspace(0, np.max(self.output_values), 10)  # 8 contour lines
        cs = ax.contour(
             phi_grid, np.degrees(theta_grid), grid_values,
             levels=contour_levels, colors='k', linewidths=0.5, alpha=0.7
        )
        #ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")  # label contour values

        cb = fig.colorbar(c, ax=ax, label="E$_d$ (eV)" if plot_type == "mean" else r"$\sigma^2$ (eV$^2$)", pad=0.1)
    

        # ax.scatter(phi, np.degrees(theta), c='black', s=0.1)
       
        plt.savefig(('tde_map.png' if plot_type == "mean" else 'tde_map_var.png') if fname is None else fname, dpi=600)
        # plt.show()


