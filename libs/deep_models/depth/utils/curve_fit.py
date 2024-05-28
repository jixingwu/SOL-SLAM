# 最小二乘曲面拟合需要预先设定拟合函数
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成一些模拟数据
np.random.seed(0)
n = 10  # 控制点数量
m = 10  # 控制点数量
degree_u = 3  # u方向的次数
degree_v = 3  # v方向的次数

u = np.linspace(0, 1, n)
v = np.linspace(0, 1, m)

# 随机生成控制点
control_points = np.random.rand(n, m, 3)

# 创建B样条曲面
bsp = BSpline(u, v, control_points, kx=degree_u, ky=degree_v)

# 生成网格点
u_vals = np.linspace(0, 1, 100)
v_vals = np.linspace(0, 1, 100)
u_grid, v_grid = np.meshgrid(u_vals, v_vals)

# 计算B样条曲面上的点
surf_points = bsp(u_grid, v_grid)

# 绘制控制点和B样条曲面
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2], c='r', marker='o')
ax.set_title('Control Points')

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(surf_points[:, :, 0], surf_points[:, :, 1], surf_points[:, :, 2], cmap='viridis', alpha=0.8)
ax.set_title('B-Spline Surface')

plt.show()
