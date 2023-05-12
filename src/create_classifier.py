import numpy as np
from PIL import Image
import os, cv2


class Dataset_Classifier():
        def train_classifer_once(self,name):
                path = os.path.join(os.getcwd()+"/data/images/"+name+"/")

                faces = []
                ids = []
                labels = []
                pictures = {}


                # Store images in a numpy format and ids of the user on the same index in imageNp and id lists

                for root,dirs,files in os.walk(path):
                        pictures = files


                for pic in pictures :
                        
                        imgpath = path+pic
                        img = Image.open(imgpath).convert('L')
                        imageNp = np.array(img, 'uint8')
                        id = int(pic.split(name)[0])
                        #names[name].append(id)
                        faces.append(imageNp)
                        ids.append(id)

                ids = np.array(ids)

                #Train and save classifier
                clf = cv2.face.LBPHFaceRecognizer_create()
                clf.train(faces, ids)
                clf.write("./data/classifiers/"+name+"_classifier.xml")

        def train_classifer_all(self):
                path = os.path.join(os.getcwd()+"/data/images")
                                
                faces = []
                ids = []
                labels = []
                label_to_id = {}
                current_id = 0 
 
                for label in os.listdir(path):
                        label_path = os.path.join(path, label)
                        
                        if os.path.isdir(label_path):
                                label_to_id[label] = current_id  
                                labels.append(label)
                                current_id += 1 
                        
                        
                        for pic in os.listdir(label_path):
                                
                                pic_path = os.path.join(label_path,pic)
                                print(pic_path)
                                
                                img = cv2.imread(pic_path)
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                
                                faces.append(gray)
                                ids.append(label_to_id[label]) 
                                
                                
                
                    
                ids = np.array(ids)
                
                clf = cv2.face.LBPHFaceRecognizer_create()
                clf.train(faces, ids)
                clf.write("./data/classifiers/trainner.xml")
                
                return labels     
        
        
        def get_labels(self) -> list:
                
                path = os.path.join(os.getcwd()+"/data/images")
                labels = []
                
                for label in os.listdir(path):
                        labels.append(label)
                
                return labels
                        
                     

                        # for pic in pictures :

                        #         imgpath = path+pic
                        #         img = Image.open(imgpath).convert('L')
                        #         imageNp = np.array(img, 'uint8')
                        #         id = int(pic.split(name)[0])
                        #         #names[name].append(id)
                        #         faces.append(imageNp)
                        #         ids.append(id)

                        # ids = np.array(ids)

                        # #Train and save classifier
                        # clf = cv2.face.LBPHFaceRecognizer_create()
                        # clf.train(faces, ids)
                        # clf.write("./data/classifiers/"+name+"_classifier.xml")
                                