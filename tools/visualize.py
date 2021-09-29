import matplotlib.pyplot as plt
import numpy as np

def decode_segmap(image, nc=2):
    label_colors = np.array([(0, 0, 0), (255, 255, 255)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def show_PreLabel_and_Image(image,pre,save_path="",tag=None):
    r"""image : size HxW """
    plt.imshow(image[0])
    plt.imshow(pre, alpha=0.5)
    if save_path!="" and tag!=None:
        plt.savefig(save_path+"/"+"%d.png" % tag)
