import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt


class VectorHeatmap:
    def __init__(self):
        self.fig, self.ax = plt.subplots()


    def pause_show(self, v, cmap='gray', interval: float=0.5):
        plt.cla()
        len_v = len(v.shape)
        w, h  = v.shape

        im = self.ax.imshow(v, cmap=cmap, vmin=v.min(), vmax=v.max())

        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # for i in range(w):
        #     for j in range(h):
        #         text = ax.text(j, i, v[i, j],
        #                     ha="center", va="center", color="gray")

        self.ax.set_title("Harvest of local farmers (in tons/year)")
        self.fig.tight_layout()
        # self.fig.colorbar(im, ax=self.ax)

        if interval < 0: plt.show()
        else:            plt.pause(interval)