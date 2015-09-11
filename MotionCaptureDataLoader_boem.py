from PCAApplier import PCAApplier


if __name__ == '__main__':
    from MotionCaptureDataLoader import MotionCaptureDataLoader
    from optparse import OptionParser, OptionValueError
    usage = "clustering with granger causality Y->X. usage: %prog [options] keyword"
    parser = OptionParser(usage)
    parser.add_option(
        "-x","--file_x",
        default= "./data/140910051-person1.trc",
        dest="datafileX",
        help="input data file X")

    parser.add_option(
        "-y","--file_y",
        default= "./data/140910051-person2.trc",
        dest="datafileY",
        help="input data file Y")

    parser.add_option(
        "-e","--embedded_frame",
        default= 20,
        dest="embeddedTime",
        type="int",
        help="create embedded vector with embedded time. numEmbeddedFrames = embeddedFrames/numSamplingFrames")

    parser.add_option(
        "-s","--sampling_frame",
        default= 1,
        dest="numSamplingFrames",
        type="int",
        help="create embedded vector with sampling frames, but all data will be used for embedded vecs")

    parser.add_option(
        "-d","--delay_frame",
        default= "1",
        dest="delayTime",
        type="int",
        help="delay time about embedded vector")

    parser.add_option(
        "-k","--num_cluster",
        default= "5",
        dest="numCluster",
        type="int",
        help="number of clusters")


    (options, args) = parser.parse_args()
    numSamplingFrames = options.numSamplingFrames
    embeddedTime = options.embeddedTime
    delayTime = options.delayTime
    numCluster = options.numCluster
    datafileX = options.datafileX
    datafileY = options.datafileY

    print "causality:" + datafileY + "->" + datafileX + "  embeddedtime %d, delaytime %d, samplingtime %d, numcluster %d"%(embeddedTime, delayTime, numSamplingFrames, numCluster)

    ##########
    loaderX = MotionCaptureDataLoader(datafileX)
    loaderY = MotionCaptureDataLoader(datafileY)

    loaderX.Load()
    loaderY.Load()

    minFrames = min(loaderX.getNumFrames(), loaderY.getNumFrames())
    loaderX.adjustNumFrames(minFrames)
    loaderY.adjustNumFrames(minFrames)



    frames_embeddedVecXt   = loaderX.generateEmbeddedVec(           1,         0, numDeleteHeadFrames = embeddedTime+delayTime-1)
    frames_embeddedVecXtkm = loaderX.generateEmbeddedVec(embeddedTime, delayTime, numSamplingFrames = numSamplingFrames)
    frames_embeddedVecYtkm = loaderY.generateEmbeddedVec(embeddedTime, delayTime, numSamplingFrames = numSamplingFrames)

    from itertools import izip
    import numpy as np
    embedded_vector = [np.r_[ np.r_[vector0, vectorX], vectorY] for vector0, vectorX, vectorY in izip(frames_embeddedVecXt, frames_embeddedVecXtkm, frames_embeddedVecYtkm)]

    pca_applier_for_classify = PCAApplier(0.9)
    frames_axes_pcaedDataMat = pca_applier_for_classify.applyPCA(embedded_vector)

    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=numCluster).fit(frames_axes_pcaedDataMat)
    frames_labels = kmeans_model.labels_


    import sys, time
    import cPickle as pickle

    pickle_dic = {"datafileX":datafileX,
                  "datafileY":datafileY,
#                  "serial_results":serial_results,
                  "serial_results":[frames_labels],
                  "embeddedTime":embeddedTime,
                  "delayTime":delayTime,
                  "numSamplingFrames":numSamplingFrames ,
                  "numCluster":numCluster}
#"data_for_movie_x":loaderX.GenerateLabeledSplitedDataForMovie(labels_frames, embeddedTime, numSamplingFrames),
                  #"data_for_movie_y":loaderY.GenerateLabeledSplitedDataForMovie(labels_frames, embeddedTime, numSamplingFrames),
                  #"data":data,


#    filename = "/data1/keisuke.kawano/results/motion_cap/" + (datafileX + datafileY).replace(".trc","_").replace("./data/","")  +"%dclusters_%dembeddedTime_%ddelayTime_%dnumSumpingFrames_"%(numCluster, embeddedTime, delayTime, numSamplingFrames) + str(time.strftime("%Y%b%d_%a_%H%M%S", time.strptime(time.ctime())))+".dat"
#    filename = "/home/kei/Desktop/kmeans_" + (datafileX + datafileY).replace(".trc","_").replace("./data/","")  +"%dclusters_%dembeddedTime_%ddelayTime_%dnumSumpingFrames_"%(numCluster, embeddedTime, delayTime, numSamplingFrames) + str(time.strftime("%Y%b%d_%a_%H%M%S", time.strptime(time.ctime())))+".dat"
    filename = "/Users/kawano/Desktop/kmeans_" + (datafileX + datafileY).replace(".trc","_").replace("./data/","")  +"%dclusters_%dembeddedTime_%ddelayTime_%dnumSumpingFrames_"%(numCluster, embeddedTime, delayTime, numSamplingFrames) + str(time.strftime("%Y%b%d_%a_%H%M%S", time.strptime(time.ctime())))+".dat"
    pickle_file = open(filename,"w")
    pickle.dump(pickle_dic, pickle_file)
    pickle_file.close()




