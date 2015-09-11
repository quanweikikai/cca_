from CausalityCalculator import CausalityCalculator
import numpy as np


# audio = np.transpose(np.loadtxt("/Users/kawano/Desktop/audio.csv", delimiter=","))
# video = np.transpose(np.loadtxt("/Users/kawano/Desktop/video.csv", delimiter=","))
# symptoms = np.transpose(np.loadtxt("/Users/kawano/Desktop/symptoms.csv", delimiter=","))

audio = np.loadtxt("/Users/kawano/Desktop/audio.csv", delimiter=",")
video = np.loadtxt("/Users/kawano/Desktop/video.csv", delimiter=",")
symptoms = np.loadtxt("/Users/kawano/Desktop/symptoms.csv", delimiter=",")

print audio
calc = CausalityCalculator(video, audio, symptoms)
calc = CausalityCalculator(audio, video, symptoms)
eigen_val, eigen_vec = calc.calcRegularizedGrangerCausality(0.99, 0.0, 0.0, 0.0)
print eigen_val

print "pcca = %f" % eigen_val ** 0.5
