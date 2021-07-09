


from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc 




def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array



def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces




def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)





trainX, trainy = load_dataset('datasets/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('datasets/validation/')
# save arrays to one file in compressed format
savez_compressed('cinema_class.npz', trainX, trainy, testX, testy)





test_1, ytest_1 = load_dataset('datasets/test/')
print(test_1.shape, ytest_1.shape)
savez_compressed('test_1.npz', test_1, ytest_1)


# # Create the embedding for the dataset 




# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]





# load the face dataset
data = load('cinema_class.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')




# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)




# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)




# save arrays to one file in compressed format
savez_compressed('cinema_class_embeddings.npz', newTrainX, trainy, newTestX, testy)




# load the face dataset
data = load('test_1.npz')
test_1, ytest_1 = data['arr_0'], data['arr_1']
print('Loaded: ', test_1.shape, ytest_1.shape)




i_test = list()
for face_pixels in test_1:
    embedding = get_embedding(model, face_pixels)
    i_test.append(embedding)
newTrainX = asarray(i_test)
print(newTrainX.shape)




savez_compressed('test_1_embeddings.npz', i_test, ytest_1)


# # Train the model 



# load dataset
data = load('cinema_class_embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# normalize input vectors





# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)





# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)





# fit model

svc = SVC(kernel='rbf',gamma=0.7,probability=True).fit(trainX, trainy)
yhat_train = svc.predict(trainX)
yhat_test = svc.predict(testX)
yhat_prob = svc.predict_proba(testX)





score_test = accuracy_score(testy, yhat_test)
print( score_test)





filename = 'finalized_model.pkl'
pickle.dump(svc, open(filename, 'wb'))


# # Calculate FFR, FAR and EER



# Load the test dataset for Both original and imposter 
data_gen = load('cinema_class.npz')
data_imposter= load('test_1.npz')
testX_imposter_faces = data_imposter['arr_0']
test_X_gen_faces = data_gen['arr_2']
print(test_X_gen_faces.shape)
print(testX_imposter_faces.shape)




# load face embeddings
data_imposter = load("test_1_embeddings.npz")
data_gen = load('cinema_class_embeddings.npz')
trainX_gen, trainy_gen, testX_gen, testy_gen = data_gen['arr_0'], data_gen['arr_1'],data_gen['arr_2'],data_gen['arr_3']
testX_imposter, testy_imposter = data_imposter['arr_0'], data_imposter['arr_1']
print(testX_gen.shape)
print(testX_imposter.shape)




# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX_gen = in_encoder.transform(trainX_gen)
testX_imposter = in_encoder.transform(testX_imposter)
testX_gen = in_encoder.transform(testX_gen)



# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy_gen)
trainy_gen = out_encoder.transform(trainy_gen)
testy_gen=out_encoder.transform(testy_gen)
# fit model
model = pickle.load(open('finalized_model.pkl', 'rb'))
model.fit(trainX_gen, trainy_gen)




# test model on imposter faces 
# a is a list where we will calculate the probability of the false accptance rate FAR
# the model will predect the face and then check the matching probablity and add it to the a list  

a = []
for i in range(100):
        random_face_pixels = testX_imposter_faces[i]
        random_face_emb = testX_imposter[i]
        random_face_class = testy_imposter[i]
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        intp = int((class_probability))
        #print(f'Predicted: {intp} %')
        a.append(intp)




# far is a list that we will save the False accaptance rate in each threshold 
 # threshold is the list of thresold and it will go from 0% to 100%       
far = []
threshold = []
for i in range(100):
        num = 0

        for x in a:
                if x>i:
                        num+=1
        #print(i,num)
        far.append(num)
        threshold.append(i)

far = np.array(far)
print('FAR: ',far)
print('-----------------------------------------------------------')




b = []
for i in range(100):
        random_face_pixels = test_X_gen_faces[i]
        random_face_emb = testX_gen[i]
        random_face_class = testy_gen[i]
        face_name = out_encoder.inverse_transform([random_face_class])
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_threshold = out_encoder.inverse_transform(yhat_class)
        if predict_threshold[0]==face_name[0]:
                intp = int((class_probability))
                #print(f'Predicted: {intp} %')
                b.append(intp)
print(b)




frr = []
for i in range(100):
        num = 0

        for x in b:
                if x<i:
                        num+=1
        #print(i,num)
        frr.append(num)


frr = np.array(frr)
print('FRR: ',frr)
print('-----------------------------------------------------------')




# calculate the EER
for i  in range(100):
        a = frr[i]
        b = far[i]
        if a == b:
                EER= a
                print('EER = ',i)




plt.plot(threshold,frr,'--b',far,'--r')
plt.plot(15,EER,'ro') 

plt.xlabel('threshold')
plt.title('FAR,FRR and EER')
plt.axis([0, 100, 0, 100])
plt.show()




threshold = np.array(threshold)




plt.plot(threshold,frr,'--b')
plt.xlabel('threshold')
plt.title('FRR')
plt.axis([0, 100, 0, 100])
plt.show()




plt.plot(threshold,far,'--r')
plt.xlabel('threshold')
plt.title('FAR')

plt.axis([5, 20, 0, 100])
plt.show()




fig, ax = plt.subplots()

ax.plot(threshold, far, 'r--', label='FAR')
ax.plot(threshold, frr, 'g--', label='FRR')
plt.xlabel('Threshold')
plt.plot(15,EER,'ro', label='EER') 


legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()








