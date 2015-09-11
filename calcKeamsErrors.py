from ArtificialDataGenerator import ArtificialDataGenerator
import numpy as np


if __name__ == '__main__':
    generator = ArtificialDataGenerator([0.33,       0.33,       0.34],
                                        ["params1", "params2", "params3"])
    generator.generate(1000)
    kmeans_results = generator.clusteringDataWithKmeans(3)
    correct_cluster_labels = generator.get_z()

    from Plotter import matchClusterLabelsForKmeans
    correct_cluster_labels,kmeans_results = matchClusterLabelsForKmeans(correct_cluster_labels, kmeans_results,3)
    num_error = np.count_nonzero(kmeans_results - correct_cluster_labels)
    print num_error
    f = open("/data1/keisuke.kawano/results/generate_model_kmeans.txt", "a")
    f.write("%s\n"%num_error)
    f.close()

