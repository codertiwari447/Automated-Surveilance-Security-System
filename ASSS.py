v
                                           #AUTONOMOUS SURVIELLANCE SECURITY SYSTEM
                                              #(Using Python, API Keras, MATLAB)


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

class Classifiers(object):

    def __init__(self,train_data,train_labels,hyperTune=True):
        self.train_data=train_data
        self.train_labels=train_labels
        self.construct_all_models(hyperTune)

    def construct_all_models(self,hyperTune):
        if hyperTune:
            #3 models KNN SCM and LR
            self.models={'SVM':[SVC(kernel='linear',probability=True),dict(C=np.arange(0.01, 2.01, 0.2))],\
                         'LogisticRegression':[lr(),dict(C=np.arange(0.1,3,0.1))],\
                         'KNN':[KNeighborsClassifier(),dict(n_neighbors=range(1, 100))],}
            for name,candidate_hyperParam in self.models.items():
                #update each classifier after training and tuning
                self.models[name] = self.train_with_hyperParamTuning(candidate_hyperParam[0],name,candidate_hyperParam[1])
            print ('\nTraining process finished\n\n\n')

    def train_with_hyperParamTuning(self,model,name,param_grid):
        #grid search method for hyper-parameter tuning
        grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
        grid.fit(self.train_data, self.train_labels)
        print(
            '\nThe best hyper-parameter for -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n'\
            .format(name, grid.best_params_, grid.best_score_))

        model = grid.best_estimator_
        train_pred = model.predict(self.train_data)
        print('{} train accuracy = {}\n'.format(name,(train_pred == self.train_labels).mean()))
        return model

    def prediction_metrics(self,test_data,test_labels,name):

        #accuracy
        print('{} test accuracy = {}\n'.format(name,(self.models[name].predict(test_data) == test_labels).mean()))

        #AUC of ROC
        prob = self.models[name].predict_proba(test_data)
        auc=roc_auc_score(test_labels.reshape(-1),prob[:,1])
        print('Classifier {} area under curve of ROC is {}\n'.format(name,auc))

        #ROC
        fpr, tpr, thresholds = roc_curve(test_labels.reshape(-1), prob[:,1], pos_label=1)
        self.roc_plot(fpr,tpr,name,auc)

    def roc_plot(self,fpr,tpr,name,auc):
        plt.figure(figsize=(20,5))
        plt.plot(fpr,tpr)
        plt.ylim([0.0,1.0])
        plt.ylim([0.0, 1.0])
        plt.title('ROC of {}     AUC: {}\nPlease close it to continue'.format(name,auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.show()

#HEIGHT/WEIGHT MATRIX

import cv2
from poscal import poscal
import numpy as np
from weight_matrix import *
from split import *
from getFeatureUV import *
from poscalNormal import *
from Classifiers import *
from labeling import *

class Feature_extractor(object):

    def __init__(self, originpics, forgpics, ab_forgpics, U, V, weigh):
        self.originpics = originpics
        self.forgpics = forgpics
        self.ab_forgpics = ab_forgpics
        self.U = U
        self.V = V
        self.weigh =weigh
        self.m = U.shape[0]
        self.n = U.shape[1]

    def getPosition(self, pics, index, style=True, mode=True):
        this_Spliter = Spliter()
        weight = self.weigh
        if mode:
            img = cv2.imread(pics[index])
            if img is None:
                img = np.zeros((self.m, self.n, 3), dtype=np.uint8)
            if style:
                ab_img = cv2.imread(self.ab_forgpics[index])
                im_s,mopho_img = poscalNormal(img,ab_img)
                splitPos = this_Spliter.split(im_s,mopho_img,weight)
                realPos, label = labeling(splitPos,ab_img)
            else:
                realPos,_ = poscal(img)
                label = np.ones(realPos.shape[0])
                mopho_img=None
        else:
            img = cv2.imread(self.forgpics[index])
            im_s,mopho_img = poscal(img)
            splitPos = this_Spliter.split(im_s,mopho_img,weight)
            realPos,label = labeling(splitPos,cv2.imread(self.ab_forgpics[index]))

        Img = cv2.imread(self.originpics[index])
        return realPos,Img,label,mopho_img

    def simgle_feature(self,pics,index,style=True,mode=True):
        realPos,_,label,_ = self.getPosition(pics,index,style,mode)

        feature = getFeaturesUV(realPos,self.U[:,:,index]* np.sqrt(self.weigh).reshape((self.m, 1)),self.V[:,:,index]* np.sqrt(self.weigh).reshape((self.m, 1)))

        return feature,label

    def get_features_and_labels(self, start, end, mode=True):
        datal = np.zeros((0,2))
        datalAb = np.zeros((0,2))
        label = np.zeros(0)
        labelAb = np.zeros(0)
        if mode:
            for i in range(start,end):
                data,labe = self.simgle_feature(self.forgpics,i)
                dataAb,labeAb = self.simgle_feature(self.ab_forgpics,i,False)
                datal = np.concatenate((datal, data), axis=0)
                datalAb = np.concatenate((datalAb, dataAb), axis=0)
                label = np.concatenate((label, labe), axis=0)
                labelAb = np.concatenate((labelAb, labeAb), axis=0)
            features = np.nan_to_num(np.concatenate((datal, datalAb), axis=0))
            labels = np.nan_to_num(np.concatenate((label, labelAb), axis=0))
        else:
            features = np.zeros((0,2))
            for i in range(start,end):
                data,_ = self.simgle_feature(self.forgpics,i,True,mode)
                features = np.nan_to_num(np.concatenate((features, data), axis=0))
            labels = None
        return features,labels

#FACIAL RECOG

from IPython.display import YouTubeVideo
import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from IPython.display import clear_output

# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()

def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()
    
class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                    cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                    cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=flags)
        return faces_coord
    
class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print(self.video.isOpened() )

    def __del__(self):
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3 
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2) 
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), 
                              (150, 150, 0), 8)

def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)






images, labels, labels_dic = collect_dataset()

rec_eig = cv2.face.EigenFaceRecognizer_create()
rec_eig.train(images, labels)

# needs at least two people 
rec_fisher = cv2.face.FisherFaceRecognizer_create()
rec_fisher.train(images, labels)

rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)

print("Models Trained Succesfully")



## testing slot
labels_dic


detector = FaceDetector("xml/frontal_face.xml")
webcam = VideoCamera(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

cv2.namedWindow("Face_Recognition", cv2.WINDOW_AUTOSIZE)

while True:
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame, True) # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            #collector = cv2.face.StandardCollector_create()
            prediction, confidence = rec_lbph.predict(face)
            #conf = collector.getMinDist()
            #pred = collector.getMinLabel()
            threshold = 140
            print("Prediction: " + labels_dic[prediction].capitalize() + "\nConfidence: " + str(round(confidence)) )
            cv2.putText(frame, labels_dic[prediction].capitalize(),
                        (faces_coord[i][0], faces_coord[i][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
        clear_output(wait = True)
        draw_rectangle(frame, faces_coord) # rectangle around face
    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
    cv2.imshow("Face_Recognition", frame) # live feed in external
    out.write(frame)
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break

del webcam

#TRAINING AI TO DETETCT WEAPONS:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array

import numpy as np
import cv2
# Image manipulations and arranging data
import os
from PIL import Image
import theano
theano.config.optimizer="None"
#Sklearn to modify the data

from sklearn.cross_validation import train_test_split
# os.chdir("provide path")

# input image dimensions
m,n = 50,50

path1="test/"
path2="train/"

classes=os.listdir(path2)
x=[]
y=[]
for fol in classes:
    print(fol)
    imgfiles=os.listdir(path2 + '/' + fol);
    for img in imgfiles:
        im=Image.open(path2+'/'+fol+'/'+img);
        im=im.convert(mode='RGB')
        imrs=im.resize((m,n))
        imrs=img_to_array(imrs)/255;
        imrs=imrs.transpose(2,0,1);
        imrs=imrs.reshape(3,m,n);
        x.append(imrs)
        y.append(fol)
        
x=np.array(x);
y=np.array(y);

batch_size=32
nb_classes=len(classes)
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)

model= Sequential()
model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'));
model.add(Convolution2D(nb_filters,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool), dim_ordering="th"));
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

nb_epoch=1
batch_size=5
history = model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
model.save('firearm_model.h5')

webcam = cv2.VideoCapture(0)
#cv2.namedWindow("Gun Detection", cv2.WINDOW_AUTOSIZE)
files=os.listdir(path1);

for i in range(15):
    #ret, img = webcam.read()
    img=files[i]
    imrs = im.resize((m,n))
    imrs=img_to_array(imrs)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    x=[]
    x.append(imrs)
    x=np.array(x);
    predictions = model.predict(x)
    print("printing model predictions:   ",predictions)
    print("predictions tpw:  ",  predictions.shape) 
    cv2.putText(img, 'Weapon detection Confidence:  ' + str(predictions),(5, 100), cv2.FONT_HERSHEY_PLAIN, 1, (66, 53, 243), 2)

    cv2.putText(img, "ESC to exit", (5, img.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
    cv2.imshow("Gun Detection", img) # live feed in external
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break
    cv2.waitKey(1000)
webcam.release()
print("printing model summary:  ", model.summary() )


