import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import RcParams
import numpy as np
import os
latex_style_times = RcParams({'font.family': 'serif',
               'font.serif': ['Times'],
               'text.usetex': True,
               })

plt.style.use(latex_style_times)
plt.rcParams['font.size'] = 12


def plot_feature(net,criterion_centerloss, plotloader,
                 device, dirname, epoch=0,plot_class_num=10,
                 maximum=500, plot_quality=150):
    plot_features = []
    plot_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(plotloader):
            inputs, targets = inputs.to(device), targets.to(device)
            embed_fea, _ = net(inputs)
            try:
                embed_fea = embed_fea.data.cpu().numpy()
                targets = targets.data.cpu().numpy()
            except:
                embed_fea = embed_fea.data.numpy()
                targets = targets.data.numpy()

            plot_features.append(embed_fea)
            plot_labels.append(targets)

    plot_features = np.concatenate(plot_features, 0)
    plot_labels = np.concatenate(plot_labels, 0)

    center_dict = criterion_centerloss.state_dict()
    centroids = center_dict['module.centers'] if isinstance(criterion_centerloss, nn.DataParallel) \
        else center_dict['centers']

    try:
        centroids = centroids.data.cpu().numpy()
    except:
        centroids = centroids.data.numpy()
    # print(centroids)
    plt.figure(figsize=(9, 9))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(plot_class_num):
        features = plot_features[plot_labels == label_idx,:]
        maximum = min(maximum, len(features)) if maximum>0 else len(features)
        plt.scatter(
            features[0:maximum, 0],
            features[0:maximum, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        # c=colors[label_idx],
        c='black',
        marker="*",
        s=5,
    )
    # currently only support 10 classes, for a good visualization.
    # change plot_class_num would lead to problems.
    legends = ['0', '1', '2', '3', '4', '5', '6', 'unknown', '8', '9']
    plt.legend(legends[0:plot_class_num]+['center'], loc='upper right', markerscale=5., fontsize = 20)

    save_name = os.path.join(dirname, 'epoch_' + str(epoch) + '.png')
    plt.savefig(save_name, bbox_inches='tight',dpi=plot_quality)
    plt.close()


