# !-*-coding:utf-8-*-
import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 実は使っているので必要
#plt.rcParams.update({"font_size": 30})


# class Plotter(object):
#    def __init__(self):
#

# def plot3dData(data,
#               clustering_answer,
#               clustering_EM_results,
#               clustering_kmeans_results,
#               clustering_oldmethod_results=None):
#    plot_colors = ["r","g","b"]
#    fig = plt.figure(figsize=(24,24))
#    ax_answer = fig.add_subplot(221, projection="3d")
#    temp_colors = [plot_colors[label] for label in clustering_answer]
#    ax_answer.scatter(data[:,0], data[:,1], data[:,2], "o", color=temp_colors)
#    ax_answer.set_title("answer")
#
#    temp_colors = [plot_colors[label] for label in clustering_EM_results]
#    ax_EM_results = fig.add_subplot(222, projection="3d")
#    ax_EM_results.scatter(data[:,0], data[:,1], data[:,2], "o", color=temp_colors)
#    ax_EM_results.set_title("MixtureOfPPCCA")
#
#    temp_colors = [plot_colors[label] for label in clustering_kmeans_results]
#    ax_kmeans_results = fig.add_subplot(223, projection="3d")
#    ax_kmeans_results.scatter(data[:,0], data[:,1], data[:,2], "o", color=temp_colors)
#    ax_kmeans_results.set_title("kmeans")
#
#    temp_colors = [plot_colors[label] for label in clustering_oldmethod_results]
#    ax_oldmethod_results = fig.add_subplot(224, projection="3d")
#    ax_oldmethod_results.scatter(data[:,0], data[:,1], data[:,2], "o", color=temp_colors)
#    ax_oldmethod_results.set_title("oldmethod")
#
#    for ii in xrange(0,360,1):
#        ax_answer.view_init(elev=ii, azim=ii)
#        ax_EM_results.view_init(elev=ii, azim=ii)
#        ax_kmeans_results.view_init(elev=ii, azim=ii)
#        ax_oldmethod_results.view_init(elev=ii, azim=ii)
#        plt.tight_layout()
#        plt.savefig("movie/%09d"%ii+".png")
#    plt.show()


def loadFile(filename, data_num):
    data = []
    for line in open(filename):
        data.append(line[1:-5].split(","))
    results = [[] for i in data]
    for i, line in enumerate(data):
        for j, element in enumerate(line):
            try:
                results[i].append(int(data[i][j]) / float(data_num))
            except:
                continue
    result_lens = []
    for result in results:
        result_lens.append(len(result))
    maxlen = max(result_lens)
    for i, result in enumerate(results):
        for j in range(maxlen - len(result)):
            results[i].append(result[-1])
    for i, result in enumerate(results):
        results[i] = result[:-1]

    return results, maxlen - 1


def plot3dData(data, cluster_indexes, plot_labels):
    import pylab as plt
    plt.rcParams['font.size'] = 30

    num_results = len(cluster_indexes)

    fig = plt.figure(figsize=(12 * num_results, 12))
    from itertools import izip
    from pylab import cm
    subplots = []
    for i, (cluster_index, plot_label) in enumerate(izip(cluster_indexes, plot_labels)):
        subplots.append(
            fig.add_subplot(101 + i + num_results * 10, projection="3d"))
        subplots[-1].scatter(data[:, 0], data[:, 1], data[:, 2], "o",
                             color=[cm.brg(ci / 3.0 + 0.1) for ci in cluster_index], lw=0)
        subplots[-1].set_title(plot_label, size=30)
        subplots[-1].set_xlabel("pca axis 1", size=30)
        subplots[-1].set_ylabel("pca axis 2", size=30)
        subplots[-1].set_zlabel("pca axis 3", size=30)

#        plt.xticks(fontsize=30)  # work on current fig
#        plt.yticks(fontsize=30)  # work on current fig
#        plt.zticks(fontsize=30)  # work on current fig


#    for ii in xrange(0, 360, 1):
    for ii in xrange(50, 70, 1):
        for subplot in subplots:
            subplot.view_init(elev=ii, azim=ii)
        plt.tight_layout()
#        plt.savefig("/data1/keisuke.kawano/results/ppccamovie/%09d" % ii + ".png")
        plt.savefig("/Users/kawano/Desktop/toy_results/ppccamovie/%09d" % ii + ".png")
# def plot3dData(data, cluster_indexes, plot_labels):
#    from mpl_toolkits.mplot3d import Axes3D
#    import pylab as plt
#
#    num_results = len(cluster_indexes)
#
#    fig = plt.figure(figsize=(12*num_results,12))
#    from itertools import izip
#    from pylab import cm
#    subplots = []
#    for i,(cluster_index,plot_label) in enumerate(izip(cluster_indexes, plot_labels)):
#        subplots.append(fig.add_subplot(101+i+num_results*10,projection="3d"))
#        subplots[-1].scatter(data[0], data[1], data[2], "o", color=[cm.brg(ci/3.0+0.1) for ci in cluster_index], lw=0)
#        subplots[-1].set_title(plot_label)
#
#    for ii in xrange(0,360,1):
#        for subplot in subplots:
#            subplot.view_init(elev=ii, azim=ii)
#        plt.tight_layout()
#        plt.savefig("/data1/keisuke.kawano/results/ppccamovie/%09d"%ii+".png")
#


def plotErrorChange(correct_cluster_labels, serial_results, filename="./test.png"):
    plt.rcParams['font.size'] = 30
    serial_error_rate = []
    for time_t, results in enumerate(serial_results):
        serial_error_rate.append(
            np.count_nonzero(results - correct_cluster_labels))
    print serial_error_rate
    plt.plot(np.array(serial_error_rate) / float(len(correct_cluster_labels)))
    plt.savefig(filename)


def plotHistConvergencedResultFromFile(filename, kmeans_filename, figname, data_num):
    error_rates, _ = loadFile(filename, data_num)
    kmeans_results = [int(line) for line in open(kmeans_filename)]
    kmeans_error_rate = np.array(kmeans_results) / float(data_num)

    import pylab as plt
    plt.clf()
    plt.rcParams['font.size'] = 15
    import matplotlib.font_manager as fm
    prop = fm.FontProperties(fname='/Library/Fonts/Osaka.ttf')

    convergence_results = [error_rate[-1] for error_rate in error_rates]
    print convergence_results
    plt.hist(convergence_results, bins=50, normed=False,
             range=(0.0, 1.0), label="proposed")
    plt.hist(kmeans_error_rate, bins=50, normed=False,
             range=(0.0, 1.0), label="previous")
    plt.xlabel(u"クラスタ非再現率", fontproperties=prop)
    plt.ylabel(u"回数", fontproperties=prop)
    plt.legend()
    plt.savefig(figname)


def matchClusterLabels(correct_labels, serial_results_labels, num_clusters):
    correct_data_index_in_cluster = []
    correct_data_index_in_cluster = [[] for i in range(num_clusters)]
    for i, correct_label in enumerate(correct_labels):
        correct_data_index_in_cluster[correct_label].append(i)

    # 最後の収束結果からクラスタ番号らしきものを逆算する
    cluster_index = []
    for cluster_i in range(num_clusters):
        result_need_in_a_cluster = np.array(
            serial_results_labels[-1])[correct_data_index_in_cluster[cluster_i]]
        cluster_index.append(int(round(np.average(result_need_in_a_cluster))))
    correct_labels = [cluster_index[label]
                      for label in correct_labels]  # 正解ラベルの順番をずらす 2->1 1->0 0->2みたいな
    return correct_labels, serial_results_labels


def matchClusterLabelsForKmeans(correct_labels, results_labels, num_clusters):
    correct_data_index_in_cluster = []
    correct_data_index_in_cluster = [[] for i in range(num_clusters)]
    for i, correct_label in enumerate(correct_labels):
        correct_data_index_in_cluster[correct_label].append(i)

    # 最後の収束結果からクラスタ番号らしきものを逆算する
    cluster_index = []
    for cluster_i in range(num_clusters):
        result_need_in_a_cluster = np.array(
            results_labels)[correct_data_index_in_cluster[cluster_i]]
        cluster_index.append(int(round(np.average(result_need_in_a_cluster))))
    correct_labels = [cluster_index[label]
                      for label in correct_labels]  # 正解ラベルの順番をずらす 2->1 1->0 0->2みたいな
    return correct_labels, results_labels


def plotSerialResultsFromFile(filename, figname, data_num):
    plt.clf()
    results, maxlen = loadFile(filename, data_num)
    ave = np.average(results, axis=0)
    std = np.std(results, axis=0)
    plt.rcParams['font.size'] = 18
    import matplotlib.font_manager as fm
    prop = fm.FontProperties(fname='/Library/Fonts/Osaka.ttf')

    plt.plot(ave, "-")
    plt.fill_between(range(maxlen), ave, ave + std, alpha=0.3, lw=0)
    plt.fill_between(range(maxlen), ave, ave - std, alpha=0.3, lw=0)
    plt.ylim(0, 1.0)
    plt.ylabel(u"クラスタ非再現率", fontproperties=prop)
    plt.xlabel(u"EMアルゴリズムの繰り返しステップ", fontproperties=prop)
#    plt.show()
    plt.savefig(figname)


if __name__ == '__main__':
    plotSerialResultsFromFile("/Users/kawano/results/toy_data/generate_model_results2.txt", "/Users/kawano/results/toy_data/generate_error_results.png", 1000)
#    plotSerialResultsFromFile("/Users/kawano/results/toy_data/gc_transitions.txt", "/Users/kawano/results/toy_data/gc_error_results.png", 3000)
    plotSerialResultsFromFile("/Users/kawano/results/toy_data/gc_transitions_dmean.txt", "/Users/kawano/results/toy_data/toy_gc_error.png", 3000)
#    plotHistConvergencedResultFromFile("/Users/kawano/results/toy_data/generate_model_results2.txt",
#                                       "/Users/kawano/results/toy_data/generate_model_kmeans.txt",
#                                       "/Users/kawano/results/toy_data/hist_generate.png", 1000)
#
#    #    plotHistConvergencedResultFromFile("/Users/kawano/results/toy_data/gc_transitions.txt",
# "/Users/kawano/results/toy_data/gc_model_kmeans.txt",
# "/Users/kawano/results/toy_data/hist_gc.png",3000)
#
#    plotHistConvergencedResultFromFile("/Users/kawano/results/toy_data/gc_transitions_dmean.txt",
#                                       "/Users/kawano/results/toy_data/gc_model_kmeans_dmean.txt",
#                                       "/Users/kawano/results/toy_data/hist_gc_dmean.png", 3000)
