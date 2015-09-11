if __name__ == '__main__':
    import sys,os
    argvs = sys.argv
    if len(argvs) != 2:
        print "usage : python %s  pickle_filename" % argvs[0]
        exit(-1)
    saveDir =  argvs[1].replace(".txt", "").replace(".dat", "")

    import pickle
    pickle_data  = pickle.load(open(argvs[1],"r"))

    serial_results    = pickle_data["serial_results"]
#    embeddedTime      = pickle_data["embeddedTime"]
#    delayTime         = pickle_data["delayTime"]
#    numSamplingFrames = pickle_data["numSamplingFrames"]
    numCluster        = pickle_data["numCluster"]
#    eigen_vals        = pickle_data["eigen_vals"]
#    eigen_vecs        = pickle_data["eigen_vecs"]
#    datafileX         =  pickle_data["datafileX"]
#    datafileY         =  pickle_data["datafileY"]

    frames_labels = serial_results[-1]
    import shutil
    labels_dirname = []
    for k in range(numCluster):
        labels_dirname.append(saveDir + "/cluster_%d"%k)
        os.mkdir(labels_dirname[-1])
    for i, label_i in enumerate(frames_labels):
        shutil.copy(saveDir+"/0/" + "%09d.png"%(i+1), labels_dirname[label_i])


