import numpy as np
from math import pi
from typing import Tuple, Optional, Callable, List




def sph_to_unit(theta, phi):
    """Convert spherical coordinates (theta colatitude, phi azimuth) to unit vectors."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)


def fibonacci_sphere(
    n: int,
    theta_range: Optional[Tuple[float, float]] = None,
    phi_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
 
    """
    Generate a set of points evenly distributed on the surface of a unit sphere.

    Parameters:
        n (int): The number of points to generate.
        theta_range (Optional[Tuple[float, float]], optional): The range of theta values to generate points in.
        phi_range (Optional[Tuple[float, float]], optional): The range of phi values to generate points in.

    Returns:
        np.ndarray: The set of points evenly distributed on the surface of a unit sphere.
    """
    if n < 50:
        UserWarning('Regression is unstable for low initial point counts -> Use init_pts greater than 50!')
        exit()
    
    # Determine the percentage of the sphere covered by given ranges
    if theta_range is not None:
        theta_min, theta_max = theta_range
    else:
        theta_min, theta_max = 0, np.pi
    if phi_range is not None:
        phi_min, phi_max = phi_range
    else:
        phi_min, phi_max = 0, 2 * np.pi
    area_fraction = (phi_max - phi_min) * (np.cos(theta_min) - np.cos(theta_max)) / (4 * np.pi)
    
    n = n * (1/area_fraction)

    # Epsilon is an offset to help counter clustering towards poles 
    epsilon = 0.5  
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    # golden angle
    ga = 2 - 2 / phi
    
    z = 1 - 2 * (i + epsilon) / n
    z = np.clip(z, -1.0, 1.0)
    r = np.sqrt(1 - z * z)
    theta =  2 * np.pi * ((i * ga) % 1)

    pts = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    theta, phi = sphere_to_polar(pts)

     # Apply masks
    mask = np.ones(len(pts), dtype=bool)
    if theta_range is not None:
        mask &= (theta >= theta_range[0]) & (theta <= theta_range[1])
    if phi_range is not None:
        mask &= (phi >= phi_range[0]) & (phi <= phi_range[1])
    

    return pts[mask]

def random_sphere_points(
    n: int,
    theta_range: tuple[float, float] = (0, np.pi),
    phi_range: tuple[float, float] = (0, 2 * np.pi)
) -> np.ndarray:
    
    """
    Generates n random points distributed on the surface of a sphere.

    Parameters:
    n (int): Number of points to generate.
    theta_range (tuple[float, float], optional): Range of theta (colatitude) values to generate. Defaults to (0, π).
    phi_range (tuple[float, float], optional): Range of phi (azimuth) values to generate. Defaults to (0, 2π).

    Returns:
    np.ndarray: Array of shape (n, 3) containing the generated points in Cartesian coordinates.
    """
    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range

    # Sample uniformly on the sphere’s surface
    u = np.random.rand(n)
    v = np.random.rand(n)
    
    # Convert to θ, φ ensuring uniform area distribution
    theta = np.arccos(np.cos(theta_min) - u * (np.cos(theta_min) - np.cos(theta_max)))
    phi = phi_min + (phi_max - phi_min) * v

    # Convert spherical to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=1)

def sphere_to_polar(points):
    # Points are (N, 3) on unit sphere
    """
    Convert points on the surface of a unit sphere to polar coordinates.

    Parameters:
        points (np.ndarray): Array of shape (N, 3) containing the points on the surface of a unit sphere.

    Returns:
        tuple: A tuple containing the polar angle (theta) and azimuthal angle (phi) of the points.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(z / r)       # polar angle [0, pi]
    phi = np.arctan2(y, x)         # azimuthal angle [-pi, pi]
    phi = np.mod(phi, 2 * np.pi)   # convert to [0, 2pi)
    return theta, phi
