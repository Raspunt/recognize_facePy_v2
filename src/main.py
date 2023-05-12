import cv2


from Detector import Detector
from create_classifier import Dataset_Classifier
from create_dataset import start_capture
from httpServer import HttpStreamer




detector = Detector()
dataset_creator = Dataset_Classifier()


# train detector
# labels = dataset_creator.train_classifer_all()
# print(labels)

labels = dataset_creator.get_labels()
print(labels)


print('traning is done')
print('enable camera')

detector.enable_camera()

http = HttpStreamer(detector,labels)
http.run()


# # start_capture("mama")

# detector.start_regognition_all(labels)
    
  





