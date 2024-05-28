import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm
import numpy as np

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig
    
def plot_depth3D(depth_map):
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    pointX, pointY, pointZ = [], [], []

    for i in tqdm(range(depth_map.shape[0])):
        for j in range(depth_map.shape[1]):
            depth = depth_map[i,j]
            if depth <= 0 or depth >= 80:
                continue
            pointX.append(-i)
            pointY.append(j)
            pointZ.append(depth)
    ax.scatter(pointX, pointY, pointZ)
    
    ax.set_xlim([-depth_map.shape[0], 0])
    ax.set_ylim([0, depth_map.shape[1]])
    ax.set_zlim([0, np.max(depth_map)])
    
    ax.set_xlabel('H')
    ax.set_ylabel('W')
    ax.set_zlabel('depth')
    
    ax.set_aspect('equalxy')

    plt.show()
    
def plot_depth2D(depth_map, XYZ='XY'):
    pointX, pointY, pointZ = [], [], []

    for i in tqdm(range(depth_map.shape[0])):
        for j in range(depth_map.shape[1]):
            depth = depth_map[i,j]
            if depth <= 0:
                continue
            pointX.append(i)
            pointY.append(j)
            pointZ.append(depth)
    if XYZ == 'XY':
        plt.scatter(pointX, pointY)
        plt.xlabel('x')
        plt.ylabel('y')
    if XYZ == 'YZ':
        plt.scatter(pointY, pointZ)
        plt.xlabel('y')
        plt.ylabel('z')
    if XYZ == 'XZ':
        plt.scatter(pointX, pointZ)
        plt.xlabel('x')
        plt.ylabel('z')
    
    plt.show()
    
def plot_mask(img, masks, colors=None, alpha=0.5) -> np.ndarray:
    if masks is None:
        return img
    
    if colors is None:
        colors = np.random.random((len(masks), 3)) * 255
    else:
        if len(colors) < len(masks):
            raise RuntimeError(
                f"colors count: {colors.shape[0]} is less than mask count: {masks.shape[0]}"
            )
    for mask, color in zip(masks, colors):
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1-alpha) + np.array(color) * alpha, img)
    
    img_seg = img.astype(np.uint8)
    
    # plt.imshow(img_seg)
    return img_seg