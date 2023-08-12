import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as mplcm
import seaborn as sns  # improves plot aesthetics

NUM_COLORS = 20
cm = plt.get_cmap('gist_rainbow')
cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
color = ['pink', 'orange', 'sienna', 'lightcoral', 'navajowhite', 'gold', 'slateblue',
         'coral', 'red', 'purple', 'royalblue', 'greenyellow', 'plum', 'bisque', 'peru', 'cyan', 'g']


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    # return limits[1] - (x - limits[0])
    return limits[1] - (x - limits[0])


def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
    return sdata


class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        # import ipdb; ipdb.set_trace()
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True,
                label="axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables, fontsize=20)
        for txt, angle in zip(text, angles):
            txt.set_rotation(angle - 90)
            
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x, 2))
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                # hack to invert grid. gridlabels aren't reversed
                grid = grid[::-1]
            gridlabel[0] = ""  # clean up origin
            l, text = ax.set_rgrids(grid, labels=gridlabel,
                          angle=angles[i], fontsize=20)
            for txt, angle in zip(text, angles):
                txt.set_rotation(angle - 90)
            # ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        # self.ax.set_prop_cycle (color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        self.ax.set_prop_cycle(color=color)

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self):
        self.ax.legend(loc=(-1, 0))


def data2vis():
    metrics = ('RMSE', 'MAE', 'R$^{2}$', 'MAPE', 'SSE')
    # firmino = [('BP', [16.2919, 5.125, 0.1647, 2.2285, 2577046.50]),
    #            ('RNN', [11.9349, 3.6941, 0.552, 1.2058, 1381963.92]),
    #            ('GRU', [11.3774, 3.6632, 0.5929, 1.2641, 1255874.39]),
    #            ('Transformer', [22.3908, 16.2609,
    #             0.1768, 44.6756, 4864090.50]),
    #            ('Informer', [14.9654, 6.6574, 0.2956, 20.3281, 2172894.03]),
    #            ('Autoformer', [15.3873, 7.0441, 0.2553, 24.6754, 2297132.26]),
    #            ('Fedformer', [24.0230, 14.4096, 0.2508, 24.0544, 5599076.35]),
    #            ('Dlinear', [20.5587, 12.46571, 0.3317, 27.4560, 4103617.24]),
    #            ('LSTM', [11.4319, 3.5501, 0.5889, 1.2662, 1267951.09]),
    #            ('BSA-LSTM', [10.9637, 3.2996, 0.6219, 1.3043, 1166218.25]),
    #            ('STL-GA-LSTM', [9.6741, 3.0479, 0.7057, 1.1707, 907987.57]),
    #            ('STL-PSO-LSTM', [9.5498, 3.0274, 0.7186, 1.2285, 868220.83]),
    #            ('STL-ZOA-LSTM', [9.3117, 2.8667, 0.7273, 1.1204, 1027783.32]),
    #            ('ALN_OOA-LSTM', [10.2234, 3.1339, 0.6713, 1.1615, 841242.69]),
    #            ('STL-BSA-LSTM', [10.2925, 3.1729, 0.6668, 1.2768, 1014033.48]),
    #            ('ALN_BSA-LSTM', [9.9899, 2.8422, 0.7458, 1.0778, 784101.15]),
    #            ('STL-ALN_BSA-LSTM',
    #             [7.3772, 2.6187, 0.8388, 0.9878, 526864.34]),
    #            ]
    firmino = [('BP', [11.8574,	5.8174,	0.5998,	7.3621,	987710.1
                       ]),
               ('RNN', [9.2939, 4.854, 0.8733, 1.5118, 606197.15]),
               ('GRU', [7.6546, 3.1851, 0.9091, 0.6523, 411210.99]),
               ('Transformer', [6.5029, 2.8311, 0.8922, 20.6603, 297157.25]),
               ('Informer', [6.4156, 2.3596, 0.8952, 11.9608, 288939.56]),
               ('Autoformer', [8.9281, 6.5552, 0.797, 76.9893, 559574.8]),
               ('Fedformer', [9.2568, 7.6565, 0.684, 2.3892, 602135.81]),
               ('Dlinear', [17.7181, 8.6815, 0.5176, 1.4907, 2205922.53]),
               ('LSTM', [8.0872, 3.2443, 0.9041, 0.7992, 434834.32]),
               ('BSA-LSTM', [7.8714, 3.3855, 0.9141, 0.5292, 459003.38]),
               ('STL-GA-LSTM', [7.8756, 3.2887, 0.9091, 0.5642, 435293.85]),
               ('STL-PSO-LSTM', [6.1661, 3.4787, 0.9243, 1.4126, 266830.53]),
               ('STL-ZOA-LSTM', [5.4175, 3.1241, 0.9253, 0.4284, 206030.99]),
               ('ALN_OOA-LSTM', [5.4659, 2.185, 0.9239, 0.4359, 209727.97]),
               ('STL-BSA-LSTM', [7.6623, 3.0134, 0.9139, 0.3833, 412037.98]),
               ('ALN_BSA-LSTM', [6.4365, 3.2439, 0.9392, 0.4827, 290743.45]),
               ('STL-ALN_BSA-LSTM',
                [4.3415, 1.5433, 0.9547, 0.325, 132320.01]),
               ]
    ranges = [(5, 30), (1, 10), (0.3, 1.0), (0, 80), (0, 3000000)]
    fig = plt.figure(figsize=(12, 9))
    radar = ComplexRadar(fig, metrics, ranges)
    for data in firmino:
        radar.plot(data[1], label=data[0])
    # fig, ax = plt.subplots(1,1)
    # ax.set_prop_cycle (color=color)
    # for data in firmino:
    #     ax.plot(data[1], label=data[0])
    # ax.legend()
    # radar.legend()
    # radar.fill(firmino, alpha=0.2)
    plt.savefig('./firmino.png')


if __name__ == '__main__':
    data2vis()
