import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras import backend as K
import tensorflow as tf

import tf2onnx

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

"""
    This script attempts to train a model which takes as input an outline of a chair, and outputs a 3d pointcloud
    of that chair.
    
    More information about creating the training data can be found in CreateTrainingData.py
    To run a model, give the filepath to the created .onnx file to the script in RunOutlineToPC_GUI.py.
    
    
    Author : Martijn Folmer
    Date : 12-08-2023
"""


class DataGenerator(tf.keras.utils.Sequence):
    """
        Creates a datagenerator for training a model using keras, so we don't have to load all the training data
        into memory at the same time.
    """
    def __init__(self, imgPaths, PCPaths, batch_size=8, shuffle=True):
        """Initialization"""
        self.imgPaths = imgPaths
        self.PCPaths = PCPaths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.imgPaths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batchImgPath = [self.imgPaths[k] for k in indexes]
        batchPCPath = [self.PCPaths[k] for k in indexes]

        X = np.asarray([cv2.imread(imgPath) for imgPath in batchImgPath])
        y = np.asarray([np.load(pcPath) for pcPath in batchPCPath])

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.imgPaths))
        if self.shuffle:
            np.random.shuffle(self.indexes)



def chamfer_distance(y_true, y_pred):

    """
        Custom loss function - Chamfer distance
        Chamfer distance is a loss function which checks the similarity between 2 unordered pointclouds by computing
        the the pairwise distance (so finding the distance for each point and its closest neighbour in the other
        pointcloud, then repeating this step for both pointclouds).
    """
    y_true = K.reshape(y_true, shape=(-1, 3))
    y_pred = K.reshape(y_pred, shape=(-1, 3))

    # Compute pairwise distance matrix
    r_true = K.sum(y_true * y_true, axis=1)
    r_true = K.reshape(r_true, [-1, 1])
    r_pred = K.sum(y_pred * y_pred, axis=1)
    r_pred = K.reshape(r_pred, [1, -1])

    D = r_true - 2 * K.dot(y_true, K.transpose(y_pred)) + r_pred

    # Compute the minimum distance for each point in y_true
    min_distance_true = K.min(D, axis=1)

    # Compute the minimum distance for each point in y_pred
    min_distance_pred = K.min(D, axis=0)

    # Compute the Chamfer Distance
    chamfer_distance = K.mean(min_distance_true) + K.mean(min_distance_pred)

    return chamfer_distance



def CreateModel(_num_points):
    """
        Create a simple keras model for this script. The input is images of size (224, 224, 3), which is standard input
        size of a MobileNet architecture. The output is (number of points in the pointcloud, 3), which is the (x,y,z)
        coordinates of each point.

        :param _num_points: The total number of points in the pointcloud
        :return: A model we can train.
    """

    # Create a basic model using Mobilenet, then remove the top
    base_model = tf.keras.applications.MobileNet()
    x = base_model.layers[-6].output

    # Dense layers for final output
    output_layer = Dense(_num_points * 3, activation=None)(x)  # Output size is num_points * 3 (X, Y, Z coordinates)
    output_layer = tf.reshape(output_layer, shape=(-1, _num_points, 3))

    # Create the model
    model = Model(inputs=base_model.input, outputs=output_layer)

    return model

# Variables that we have to put in
pathToFoldData = f'Path/To/Where/We/Stored/TrainingData'        # Where we saved our training data created in CreateTrainingData.py
modelNameToSave = f'OutLineToPC_chair_1024_all'                 # The name of the model we want to save it under
pathToSaveLocation = 'ModelsOutlineToPC'                        # where we want to save the trained model
pathToSampleLocation = 'test_img'                               # where we save the images we do
batch_size = 8                                                  # the batch size we train the model under


# Load the models
all_fold = [pathToFoldData + f"/{foldName}" for foldName in os.listdir(pathToFoldData)]
allImgPath = []
for fold in all_fold:
    allImgPath.extend([fold + f"/{fileName}" for fileName in os.listdir(fold) if '.png' in fileName])
AllPCPath = [imgPath[:-4] + '.npy' for imgPath in allImgPath]


LoadTheModel = False
if LoadTheModel:
  model = tf.keras.models.load_model(f'{pathToSaveLocation}/{modelNameToSave}', custom_objects={'chamfer_distance' :chamfer_distance})
  model.summary()
else:
    model = CreateModel()
    model.summary()
    # Compile the model with the Chamfer Distance loss
    model.compile(optimizer='adam', loss=chamfer_distance)


# Remove all images we may have made during a previous testing round
if not os.path.exists(f'{pathToSampleLocation}'): os.mkdir(f'{pathToSampleLocation}')
all_files = [f'{pathToSampleLocation}/{filename}' for filename in os.listdir(f'{pathToSampleLocation}')]
for file in all_files:
    os.remove(file)

# Split our training data into Training and Testing datasets and initialize our Datagenerators
trainImg, testImg, trainPC, testPC = train_test_split(allImgPath, AllPCPath, test_size = 0.2, shuffle=True)
TrainGenerator = DataGenerator(trainImg, trainPC, batch_size, True)
TestGenerator = DataGenerator(testImg, testPC, batch_size, True)

# Train the model
model.fit(TrainGenerator, validation_data = TestGenerator, epochs=500, batch_size=batch_size)

# Save both the keras model and the .onnx model
model.save(f'{pathToSaveLocation}/{modelNameToSave}')
onnx_name = f'{pathToSaveLocation}/{modelNameToSave}.onnx'
(onnx_model_proto, storage) = tf2onnx.convert.from_keras(model)
with open(onnx_name, "wb") as f:
    f.write(onnx_model_proto.SerializeToString())


# VISUALISATION : The following part of the code is used to run several Data samples from our Train and Test generator
# through the model we just trained, and visualising the output so we can check how wel the model is working.
# Please note that in case of overfitting, the 'Train' images will seem much more accurate that the 'Test' images,
# so pay extra attention to how the 'Test' images perform.

def plot_point_cloud(point_cloud, title):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-112, 112])
    ax.set_ylim([-112, 112])
    ax.set_zlim([-112, 112])
    ax.set_title(title)
    ax.view_init(elev=90, azim=-90)

    plt.savefig('temp.png', bbox_inches='tight')
    plt.close()

    img = cv2.imread('temp.png')
    img = cv2.resize(img, (224, 224))
    os.remove('temp.png')

    return img

kn = 0
for i in range(5):
    Data = TrainGenerator.__getitem__(i)
    output = model.predict(Data[0])

    for i_img, (img, groundTruthPC, predictedPC) in enumerate(zip(Data[0], Data[1], output)):

        img_pc_groundTruth = plot_point_cloud(groundTruthPC, 'Groundtruth')
        img_pc_predicted = plot_point_cloud(predictedPC, 'Predicted')

        imgTot = np.concatenate([img, img_pc_groundTruth, img_pc_predicted], axis=1)

        cv2.imwrite(f'{pathToSampleLocation}/img_{kn}_train_model_all.png', imgTot)
        kn += 1
        print(f"We are at model : all : train {kn}")

for i in range(2):
    Data = TestGenerator.__getitem__(i)
    output = model.predict(Data[0])

    for i_img, (img, groundTruthPC, predictedPC) in enumerate(zip(Data[0], Data[1], output)):
        img_pc_groundTruth = plot_point_cloud(groundTruthPC, 'Groundtruth')
        img_pc_predicted = plot_point_cloud(predictedPC, 'Predicted')

        imgTot = np.concatenate([img, img_pc_groundTruth, img_pc_predicted], axis=1)

        cv2.imwrite(f'{pathToSampleLocation}/img_{kn}_test_model_all.png', imgTot)
        kn += 1
        print(f"We are at model : all : test {kn}")

