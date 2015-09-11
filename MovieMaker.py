# -*-coding:utf-8-*-

from itertools import izip
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib


class MovieMaker(object):

    """
    # 始めにキャンバスの数を指定して、addXXX関数でcanvasを順に埋めていく
    # 最後に draw or savefigs or record でデータを出力する
    # 大量のデータを一気にムービーにしようとすると固まる(?)ので、savefigs を使って一旦画像に出力した後でムービーを作ったほうが安全(?)
    """

    def __init__(self, num_canvas, num_canvas_horizontal=3):
        self.canvas_index = 1
        self.num_canvas_x = (num_canvas - 1) / num_canvas_horizontal + 1
        self.num_canvas_y = num_canvas_horizontal

        self.figure = plt.figure(
            figsize=(6 * self.num_canvas_y, 6 * self.num_canvas_x), dpi=100)

        self.updateFunctions = []
        self.num_frameses = []

    def draw(self):
        animations = []
        for updateFunc, num_frames in izip(self.updateFunctions, self.num_frameses):
            animations.append(animation.FuncAnimation(
                self.figure, updateFunc, num_frames, interval=10, blit=False))
        plt.show()

    def savefigs(self, dirname, label_i):
        import os
        label_dir = dirname + "/%d" % label_i
        os.mkdir(label_dir)
        for time in range(self.num_frameses[0]):  # 本当は一番短いムービーに合わせたりすべきかも
            for updateFunc in self.updateFunctions:
                updateFunc(time)
            # ffmpegではファイルは1番から
#            plt.savefig(label_dir + "/%09d" % (time + 1) + ".png")
            matplotlib.use('Agg')
            plt.savefig(label_dir + "/%09d" % (time + 1) + ".eps")
    #        plt.savefig(label_dir + "/%09d"%(time + 1) + ".jpg")

    def record(self, filename, fps, dpi=100, bitrate=720):
        def updateFunctions(time):
            for updateFunc in self.updateFunctions:
                updateFunc(time)

        ani = animation.FuncAnimation(
            self.figure, updateFunctions, self.num_frameses[0], interval=10, blit=False)
        ani.save(filename, fps=fps, writer="ffmpeg", dpi=dpi, bitrate=bitrate)

    """
    これ以降は設定した枠にどんなMovie埋めるかという関数
    """

    def addImageMovie(self, images, frames_in_label, embedded_time, numSamplingFrames=1):
        axes = self.figure.add_subplot(
            self.num_canvas_x, self.num_canvas_y, self.canvas_index)

        frames_to_record = []
        for begin_frame in frames_in_label:
            for i in range(embedded_time):
                frames_to_record.append(begin_frame + i)

        def updateFunc(time):
            axes.cla()
            axes.imshow(images[frames_to_record[time]])

        self.updateFunctions.append(updateFunc)
        self.num_frameses.append(len(frames_to_record))
        self.canvas_index += 1

    def addPCAMovie(self, frames_in_label, frames_clusterLabelsMat, frames_axes_pcaedDataMat, embedded_time, numSamplingFrames=1, alpha=0.2, frames_labels=[]):
        axes = self.figure.add_subplot(
            self.num_canvas_x, self.num_canvas_y, self.canvas_index)
        axes_frames_pcaedDataMat = frames_axes_pcaedDataMat.T
        axes.scatter(axes_frames_pcaedDataMat[0], axes_frames_pcaedDataMat[1],
                     c=frames_clusterLabelsMat, alpha=0.5, marker=".")
        axes.set_title("")
        plt.tight_layout()  # いらないかも

        def updateFunc(time):
            axes.cla()
            axes.scatter(axes_frames_pcaedDataMat[0], axes_frames_pcaedDataMat[1],
                         c=frames_clusterLabelsMat, alpha=alpha,
                         linewidth='0', marker=".")

            axes.scatter(axes_frames_pcaedDataMat[0][frames_in_label[time / (embedded_time / numSamplingFrames)]],
                         axes_frames_pcaedDataMat[1][
                             frames_in_label[time / (embedded_time / numSamplingFrames)]],
                         c="black", alpha=1, marker=".")
#            axes.set_title("eigen_val = %f"%(frames_clusterLabelsMat[time]/(embedded_time/numSamplingFrames)))
            axes.set_title("cluster = %d,  granger_causality = %f" % (
                frames_labels[time], frames_clusterLabelsMat[time]))

        self.updateFunctions.append(updateFunc)
        self.num_frameses.append(
            len(frames_in_label) * embedded_time / numSamplingFrames)
        self.canvas_index += 1

    def addSomeSticksMovie(self, frames_jointsxyz_positions, limbs_jointsIndecies, embeddedTime, numSamplingFrames=1, person_label=""):
        axes = self.figure.add_subplot(
            self.num_canvas_x, self.num_canvas_y, self.canvas_index, projection="3d")

        axes.set_xlabel('x[mm]')
        axes.set_ylabel('y[mm]')
        axes.set_zlabel('z[mm]')
        axes.set_xlim(-400, 400)
        axes.set_ylim(-400, 400)
        axes.set_zlim(-400, 400)
        plots = []
        title = plt.title("")
        num_data = len(frames_jointsxyz_positions)
        for temp_jointsIndecies in limbs_jointsIndecies:
            jointsIndecies = np.array(temp_jointsIndecies)
            plots.append(axes.plot(
                frames_jointsxyz_positions[0, jointsIndecies * 3 + 0],
                frames_jointsxyz_positions[0, jointsIndecies * 3 + 1],
                frames_jointsxyz_positions[0, jointsIndecies * 3 + 2],
                "-o")[0])
        plt.tight_layout()

        axes.view_init(azim=135, elev=20)

        #####
        def updateFunc(time):
            for limb_i, temp_jointsIndecies in enumerate(limbs_jointsIndecies):
                jointsIndecies = np.array(temp_jointsIndecies)
                plots[limb_i].set_xdata(
                    frames_jointsxyz_positions[time, jointsIndecies * 3 + 0])
                plots[limb_i].set_ydata(
                    frames_jointsxyz_positions[time, jointsIndecies * 3 + 1])
                plots[limb_i].set_3d_properties(
                    frames_jointsxyz_positions[time, jointsIndecies * 3 + 2])
#            title.set_text("data = %d/%d, time=%d/%d"%(time*numSamplingFrames/embeddedTime, num_data, time%(embeddedTime/numSamplingFrames), embeddedTime/numSamplingFrames))
            title.set_text(person_label + "    frame:%d/%d" %
                           (time * numSamplingFrames / embeddedTime, num_data))

        self.updateFunctions.append(updateFunc)
        self.num_frameses.append(len(frames_jointsxyz_positions))
        self.canvas_index += 1

    def addAlphaSomeSticksMovie(self, start_frames, frames_jointsxyz_positions, limbs_jointsIndecies, embeddedTime, numSamplingFrames=1, person_label=""):
        plt.rcParams['font.size'] = 18

        axes = self.figure.add_subplot(self.num_canvas_x, self.num_canvas_y, self.canvas_index, projection="3d")
        axes.set_xlabel('x[mm]')
        axes.set_ylabel('y[mm]')
        axes.set_zlabel('z[mm]')

#        axes.set_xticks([-400, -200, 0, 200, 400], 'x[mm]')
#        axes.set_yticks([-400, -200, 0, 200, 400], 'y[mm]')
#        plt.zticks([-400, -200, 0, 200, 400])
        axes.set_xlim(-400, 400)
        axes.set_ylim(-400, 400)
        axes.set_zlim(-400, 400)
        start_frames_plots = [[] for i in range(len(start_frames))]
        title = plt.title("")

        axes.view_init(azim=135, elev=20)

        colors = ["b", "g", "r"]

        for start_frame_i in range(len(start_frames)):
            for color_i, temp_jointsIndecies in enumerate(limbs_jointsIndecies):
                jointsIndecies = np.array(temp_jointsIndecies)
                start_frames_plots[start_frame_i].append(axes.plot(
                    frames_jointsxyz_positions[0, jointsIndecies * 3 + 0],
                    frames_jointsxyz_positions[0, jointsIndecies * 3 + 1],
                    frames_jointsxyz_positions[0, jointsIndecies * 3 + 2],
                    "-", alpha=0.3, c=colors[color_i])[0])

        def updateFunc(time):
            for start_frame_i, start_frame in enumerate(start_frames):
                for limb_i, temp_jointsIndecies in enumerate(limbs_jointsIndecies):
                    jointsIndecies = np.array(temp_jointsIndecies)
                    start_frames_plots[start_frame_i][limb_i].set_xdata(
                        frames_jointsxyz_positions[start_frame + time, jointsIndecies * 3 + 0])
                    start_frames_plots[start_frame_i][limb_i].set_ydata(
                        frames_jointsxyz_positions[start_frame + time, jointsIndecies * 3 + 1])
                    start_frames_plots[start_frame_i][limb_i].set_3d_properties(
                        frames_jointsxyz_positions[start_frame + time, jointsIndecies * 3 + 2])
                title.set_text(person_label + "    frame:%d/%d" %
                               (time, embeddedTime))
        self.updateFunctions.append(updateFunc)
        self.num_frameses.append(embeddedTime)
        self.canvas_index += 1


def SplitData(data, frames_labels):
    numLabels = max(frames_labels) + 1
    labels_frames_embeddedVecYtkm = [[] for i in range(numLabels)]
    labels_frames_embeddedVecXtkm = [[] for i in range(numLabels)]
    labels_frames_embeddedVecXt = [[] for i in range(numLabels)]
    for frame_i, temp_data in enumerate(data):
        labels_frames_embeddedVecXtkm[frames_labels[frame_i]].append(
            np.array(temp_data[0]).flatten())
        labels_frames_embeddedVecYtkm[frames_labels[frame_i]].append(
            np.array(temp_data[1]).flatten())
        labels_frames_embeddedVecXt[frames_labels[frame_i]].append(
            np.array(temp_data[2]).flatten())
    return labels_frames_embeddedVecYtkm, labels_frames_embeddedVecXtkm, labels_frames_embeddedVecXt


if __name__ == '__main__':

    def get_pcaedData(data, n_components_):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components_)
        inputs_data = []
        for temp_data in data:
            temp_inputs = []
            for temp in temp_data:
                temp_inputs += np.array(temp.reshape(-1,)).tolist()[0]
            inputs_data.append(temp_inputs)
        return pca.fit_transform(np.array(inputs_data))

    import sys
    argvs = sys.argv
    if len(argvs) != 2:
        print "usage : python %s  pickle_filename" % argvs[0]
        exit(-1)
    saveDir = argvs[1].replace(".txt", "").replace(".dat", "")
    import os
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    import pickle
    pickle_data = pickle.load(open(argvs[1], "r"))

#    labels_joints_data_x = pickle_data["data_for_movie_x"]
#    labels_joints_data_y = pickle_data["data_for_movie_y"]
#    data = pickle_data["data"]
    serial_results = pickle_data["serial_results"]
    embeddedTime = pickle_data["embeddedTime"]
    delayTime = pickle_data["delayTime"]
    numSamplingFrames = pickle_data["numSamplingFrames"]
    numCluster = pickle_data["numCluster"]
#    eigen_vals        = pickle_data["eigen_vals"]
#    eigen_vecs        = pickle_data["eigen_vecs"]
    datafileX = pickle_data["datafileX"]
    datafileY = pickle_data["datafileY"]
    #############
    from MotionCaptureDataLoader import MotionCaptureDataLoader
    loaderX = MotionCaptureDataLoader(datafileX)
    loaderY = MotionCaptureDataLoader(datafileY)

    loaderX.Load()
    loaderY.Load()

    minFrames = min(loaderX.getNumFrames(), loaderY.getNumFrames())
    loaderX.adjustNumFrames(minFrames)
    loaderY.adjustNumFrames(minFrames)

    ##
    frames_embeddedVecXt = loaderX.generateEmbeddedVec(1, 0, numDeleteHeadFrames=embeddedTime + delayTime - 1)
    frames_embeddedVecXtkm = loaderX.generateEmbeddedVec(embeddedTime, delayTime, numSamplingFrames=numSamplingFrames)
    frames_embeddedVecYtkm = loaderY.generateEmbeddedVec(embeddedTime, delayTime, numSamplingFrames=numSamplingFrames)
    data = [[np.matrix(embeddedVecXt).T, np.matrix(embeddedVecYtkm).T, np.matrix(embeddedVecXtkm).T]
            for i, [embeddedVecXt, embeddedVecYtkm, embeddedVecXtkm] in enumerate(izip(frames_embeddedVecXt, frames_embeddedVecYtkm, frames_embeddedVecXtkm))]
    pcaed_data = get_pcaedData(data, 2)
    ##

    frames_positions_x = loaderX.GenerateDataForMovie(embeddedTime + delayTime - 1)
    frames_positions_y = loaderY.GenerateDataForMovie(embeddedTime + delayTime - 1)
    #############

    frames_labels = serial_results[-1]
    labels_frames_embeddedVecXt, labels_frames_embeddedVecXtkm, labels_frames_embeddedVecYtkm = SplitData(data, frames_labels)

    from PCAApplier import PCAApplier
    pcaContribution = 0.99
    pca = PCAApplier(pcaContribution)
    labels_pcaed_Xt = []
    labels_pcaed_Xtkm = []
    labels_pcaed_Ytkm = []
    for i in range(numCluster):
        labels_pcaed_Xt.append(pca.applyPCA(labels_frames_embeddedVecXt[i]))
        labels_pcaed_Xtkm.append(pca.applyPCA(labels_frames_embeddedVecXtkm[i]))
        labels_pcaed_Ytkm.append(pca.applyPCA(labels_frames_embeddedVecYtkm[i]))

    # calc Causalities
    from CausalityCalculator import CausalityCalculator
#    from SurrogatingTester import SurrogatingTester

#    num_try = 100
#    tester = SurrogatingTester(num_try, numCluster, labels_frames_embeddedVecXt, labels_frames_embeddedVecXtkm, labels_frames_embeddedVecYtkm)

    causalities = []
    for k in range(numCluster):
        print np.array(labels_frames_embeddedVecXt[k]).shape
        calclator = CausalityCalculator(labels_frames_embeddedVecXt[k],
                                        # 2番めの引数の影響を除く
                                        labels_frames_embeddedVecXtkm[k],
                                        labels_frames_embeddedVecYtkm[k])
        causalities.append(calclator.calcRegularizedGrangerCausality(0.99, 0.0001, 0.0001, 0.0001))
#    surrogated_causalities = tester.compare(causalities)
#    print tester.get()

#    frames_vals = [surrogated_causalities[label_i] for frame_i, label_i in enumerate(frames_labels)]
    frames_vals = [causalities[label_i] for frame_i, label_i in enumerate(frames_labels)]
#    frames_vals = [eigen_vals[label_i][0] for frame_i, label_i in enumerate(frames_labels)]

    maker = MovieMaker(num_canvas=2, num_canvas_horizontal=2)
    maker.addSomeSticksMovie(frames_positions_x, [[0, 1, 2, 3], [3, 6, 5, 4], [2, 6]], 1, 1, "X")
    maker.addSomeSticksMovie(frames_positions_y, [[0, 1, 2, 3], [3, 6, 5, 4], [2, 6]], 1, 1, "Y")
#    maker.addPCAMovie(range(len(frames_positions_x)),
#                      frames_vals, pcaed_data, 1, 1, frames_labels=frames_labels)
    maker.savefigs(saveDir, 98)

#    f = open(saveDir + "/causalities.txt", "w")
#    for i, causality in enumerate(causalities):
#    for i, causality in enumerate(surrogated_causalities):
#        f.write("label = %d, \t" % i + str(causality) + "\n")
#    f.close()

#    temp_label = frames_labels[0]
#    temp_frame = 0
#    labels_start_frames = [[] for i in range(numCluster)]
#    for frame_i, label in enumerate(frames_labels):
#        if temp_label == label:
#            continue
#        start_frame = (frame_i + temp_frame) / 2 - embeddedTime - delayTime
#        if start_frame < 0:
#            continue
#        labels_start_frames[temp_label].append(start_frame)
#        temp_label = label
#        temp_frame = frame_i
#
#    for label_i in range(numCluster):
#        maker = MovieMaker(num_canvas=2, num_canvas_horizontal=2)
#        maker.addAlphaSomeSticksMovie(labels_start_frames[label_i],
#                                      frames_positions_x,
#                                      [[0, 1, 2, 3], [3, 6, 5, 4], [2, 6]], embeddedTime, 1, "person_X")
#        maker.addAlphaSomeSticksMovie(labels_start_frames[label_i],
#                                      frames_positions_y,
#                                      [[0, 1, 2, 3], [3, 6, 5, 4], [2, 6]], embeddedTime, 1, "person_Y")
#        maker.savefigs(saveDir, label_i)
#
#
#
#    labels_frames = [[] for i in range(numCluster)]
#    for i, label in enumerate(serial_results[-1]):
#        labels_frames[label].append(i)
#
#    ##
#    frames_vals = [eigen_vals[label_i][0] for frame_i,label_i in enumerate(frames_labels)] #それぞれのクラスタの最大の固有値で色付け
#    ##
#
#    def makeMovie(label_i):
#        joints_data_x = labels_joints_data_x[label_i]
#        joints_data_y = labels_joints_data_y[label_i]
#        movie = MovieMaker(num_canvas = 3, num_canvas_horizontal = 3)
#        movie.addSomeSticksMovie(joints_data_x,
#                                 [[0,1,2,3],[3,6,5,4],[2,6]],embeddedTime, numSamplingFrames)
#        movie.addSomeSticksMovie(joints_data_y,
#                                 [[0,1,2,3],[3,6,5,4],[2,6]],embeddedTime, numSamplingFrames)
#        movie.addPCAMovie(labels_frames[label_i], serial_results[-1], pcaed_data, embeddedTime, numSamplingFrames)
#        movie.addPCAMovie(labels_frames[label_i], frames_vals, pcaed_data, embeddedTime, numSamplingFrames)
#        movie.addPCAMovie(labels_frames[label_i], frames_causalities, pcaed_data, embeddedTime, numSamplingFrames)
#        movie.savefigs(saveDir, label_i)
#
#    import multiprocessing as mp
#    processes = []
#    for i in range(numCluster):
#        p = mp.Process(target = makeMovie, args=(i,))
#        processes.append(p)
#        p.start()
#    for p in processes:
#        p.join()
