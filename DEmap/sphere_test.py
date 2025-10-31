import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from DEmap.sphere_utils import *
# Example: generate points in a wedge
pts = fibonacci_sphere(
    n=200,
    theta_range=(0, np.pi),   # colatitude 0..45 degrees
    phi_range=(0, np.pi*2)      # azimuth 0..90 degrees
)

# 3D plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='blue', s=20)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,1])  # Equal aspect ratio

plt.show()

