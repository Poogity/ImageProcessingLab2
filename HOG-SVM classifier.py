import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from patchify import patchify
import pickle
from sklearn import svm
from sklearn.metrics import confusion_matrix

class Data():
    def __init__(self, data):
        self.data = data
        
def saveData(data,filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def loadData(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
        
bin_n = 9 # Number of bins
def myHog(img, patchSize):
    hogDescriptor = np.empty(0)
    img= cv2.resize(img, (32,32))
    patches = patchify(img, (patchSize,patchSize), step=patchSize)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i][j]
            
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
            g, theta = cv2.cartToPolar(gx, gy)
            
            bins = np.int32(bin_n*theta/(2*np.pi))    # quantizing binvalues in (0...9)
            
            bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
            mag_cells =g[:10,:10],g[10:,:10],g[:10,10:],g[10:,10:]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells,mag_cells)]
            hist = np.hstack(hists)     # hist is a 64 bit vector
            hogDescriptor = np.hstack((hogDescriptor,hist))
    
        
    return hogDescriptor[:36*int(32/patchSize)**2]

train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset= train_dataset,batch_size = 1,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset,batch_size = 1, shuffle=False)


examples = iter(train_loader)
samples, labels = examples.next()

def createData():
    patchSize = 8
    trainingData = np.empty((60000,36*int(32/patchSize)**2))
    testingData = np.empty((10000,36*int(32/patchSize)**2))
    responses = np.empty(60000)
    testResponses = np.empty(10000)
    for i, (images, labels) in enumerate(train_loader):
        img = np.array(images[0][0])
        hog = np.reshape(myHog(img,patchSize), (1,36*int(32/patchSize)**2))
        trainingData[i] = hog
        responses[i] = labels
        if (i+1) % 2000 == 0:
            print (f'HOG for train image [{i+1}/{60000} created]')
    
    saveData(trainingData, "trainingData.pickle")
    saveData(responses, "trainResponses.pickle")
    
    
    for i, (images, labels) in enumerate(test_loader):
        img = np.array(images[0][0])
        hog = np.reshape(myHog(img,patchSize), (1,36*int(32/patchSize)**2))
        testingData[i] = hog
        testResponses[i] = labels
        if (i+1) % 2000 == 0:
            print (f'HOG for test image [{i+1}/{10000} created]')
            
    saveData(testingData, "testingData.pickle")
    saveData(testResponses, "testResponses.pickle")

#comment the below function if data is already saved
createData()

trainingData = np.array(loadData("trainingData.pickle"))
trainResponses = np.array(loadData("trainResponses.pickle"))
testingData = np.array(loadData("testingData.pickle"))
testResponses = np.array(loadData("testResponses.pickle"))

#comment the below if classifier already saved
#'''
# Create an SVM classifier
classifier = svm.SVC()
print("Training the SVM classifier...")
# Train the model
classifier.fit(trainingData, trainResponses)
print("Classifier Trained!")
saveData(classifier, "classifier.pickle")
#'''
classifier = loadData("classifier.pickle")
# Predict the labels for the test set
print("Classifying the test data...")
y_pred = classifier.predict(testingData)
# Evaluate the accuracy of the model
print("Classification finished:\n")

accuracy = classifier.score(testingData, testResponses)
print("Accuracy:", accuracy)


# Assuming X_test contains the test data features and y_test contains the corresponding true labels
# Predict the labels for the test set
# Calculate the confusion matrix
cm = confusion_matrix(testResponses, y_pred, normalize = 'true')
