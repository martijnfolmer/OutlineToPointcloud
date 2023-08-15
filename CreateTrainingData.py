import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import trimesh
import math
import os
import random

import tensorflow as tf

"""
    This .py file is used to create the training data with which we train our Outline -> pointcloud model. 
    
    We want the input of our model to be a (224x224x3) image of the outline of a chair. 
    We want the output of our model to be a 3D pointcloud of a chair which fits inside of the outline of the chair.
    
    We achieve this by taking a bunch of 3D models of chairs from a public dataset (more info about the data can be found
    here https://3dvision.princeton.edu/projects/2014/3DShapeNets/). We use these models to create both an outline
    from a certain angle as a pointcloud.
    
    The models we took are all save under an .off file format, which gives information about a 3d models vertices,
    faces and edges. Using this information, a 2D representation can be drawn on an image, after which we can
    use canny edge detection to create the outline. The pointclouds can be generated using the trimesh library.
    
    The training data is augmented by adding noise to the pointclouds (+- a certain amount to each coordinate), rotating
    the model around the z-axis (so we see if from different angles).
   
    Please note : this is a horribly inefficient way to generate this data, and many improvements can be done in terms
    of memory and not constantly saving and rereading the same files. However, this is just a side-project for a 
    proof of concept, so we only had to generate the data once (which was achieved by leaving the generator running
    overnight). I encourage you to take these scripts and optimize them to your needs ;-)
    
    
    Author : Martijn Folmer
    Data : 12-08-2023


"""

def read_off(offFilePath):

    """
    Take a .off file, and read all the information about vertices and faces

    :param offFilePath: The location of the .off file we want to read.
    :return: a list with all the vertices, a list with all the faces, and the number of vertices, faces and edges.
    Please note that in this project, all .off files had no edges, so n_edges was always zero.
    Also, frustratingly, .off files have a nasty habit of not being parsed
    """

    file = open(offFilePath, 'r')
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_edges = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    # edges = [[int(s) for s in file.readline().strip().split(' ')] for i_edges in range(n_edges)]
    file.close()
    return verts, faces, n_verts, n_faces, n_edges


def get_pcloud(file, nPoints):
    """
    Use the trimesh library to generate a pointcloud with nPoints amount of numbers

    :param file:  The path to the .off file
    :param nPoints: The amount of points we want the pointcloud to consist of
    :return: The pointcloud (list of [x,y,z] coordinates)
    """

    pcloud = trimesh.load(file).sample(nPoints)
    return pcloud


def save_OFF_file(filePath, verts, faces, num_edges):
    """
    Create a .off file based on generated vertices and faces

    :param filePath: The location of where we should save the .off file
    :param verts: List of vertices
    :param faces: List of faces.
    :param num_edges: In this case, this should always be 0, as our .off files don't have any edges in them
    :return: --
    """

    n_verts, n_faces, n_edges = len(verts), len(faces), num_edges

    file = open(filePath, "w")
    file.write("OFF\n")
    file.write(f"{n_verts} {n_faces} {n_edges}\n")

    # write vertices
    for vert_cur in verts:
        file.write(f"{vert_cur[0]} {vert_cur[1]} {vert_cur[2]}\n")

    # write faces
    for i_face, face_cur in enumerate(faces):
        face_string = ""
        for num in face_cur:
            face_string += f"{num} "
        face_string = face_string[:-1] + "\n"
        file.write(face_string)

    file.close()


def getLim(xcur, ycur, zcur):
    """
    Find the limits of the x, y and z coordinates

    :param xcur: a list of the x-coordinates
    :param ycur: a list of the y-coordinates
    :param zcur: a list of the z-coordinates
    :return: The maximum limits
    """

    minx, maxx = min(xcur), max(xcur)
    miny, maxy = min(ycur), max(ycur)
    minz, maxz = min(zcur), max(zcur)
    minlim = min(minx, miny, minz)
    maxlim = max(maxx, maxy, maxz)
    diff = maxlim - minlim
    return minlim, maxlim, diff

def plot_point_cloud(xcur, ycur, zcur, title):
    """
    Visualisation function, plot the pointcloud that we have created

    :param xcur: a list of x-coordinates
    :param ycur: a list of y-coordinates
    :param zcur: a list of z-coordinates
    :param title: The title we want to put on top of the graph

    :return: an cv2 compatible image we can display
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xcur, ycur, zcur, s=10)
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

    os.remove('temp.png')
    return img




def drawPolygons(img, x, y, list_of_idx, color = (255, 0, 0)):

    # idx = [idx1, idx2, idx3, ..., idxn]
    # x = list of x coordinates
    # y = list of y coordinates

    pts = np.asarray([[int(x[idxc]),int(y[idxc])] for idxc in list_of_idx])
    cv2.fillPoly(img, pts=[pts], color=color)

    return img



def rotation_matrix_x(theta_x):
    """
    Create a 3D rotation matrix for rotation around the x-axis.

    :param theta_x: Rotation angle in radians.
    :return: The 3x3 rotation matrix.
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

def rotation_matrix_y(theta_y):
    """
    Create a 3D rotation matrix for rotation around the y-axis.

    :param theta_y: Rotation angle in radians.
    :return: The 3x3 rotation matrix.
    """
    return np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

def rotation_matrix_z(theta_z):
    """
    Create a 3D rotation matrix for rotation around the z-axis.

    :param theta_z: Rotation angle in radians.
    :return: The 3x3 rotation matrix.
    """
    return np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

# Step 4: Combine the rotation matrices and apply them to the point

def rotate_point_3d(point, theta_x, theta_y, theta_z):
    """
    Rotate a 3D point around the global axes (x, y, z) by angles (theta_x, theta_y, theta_z).

    :param point: 1D numpy array representing the 3D point [x, y, z].
    :param theta_x: Rotation angle around the x-axis in radians.
    :param theta_y: Rotation angle around the y-axis in radians.
    :param theta_z: Rotation angle around the z-axis in radians.
    :return: The rotated 3D point as a 1D numpy array.
    """
    R_x = rotation_matrix_x(theta_x)
    R_y = rotation_matrix_y(theta_y)
    R_z = rotation_matrix_z(theta_z)

    # Combine the rotation matrices (order matters: first X, then Y, and finally Z)
    R_combined = np.dot(R_z, np.dot(R_y, R_x))

    # Rotate the point by multiplying with the combined rotation matrix
    rotated_point = np.dot(R_combined, point)

    return rotated_point

def getContour(img):
    """
    Get the outline of our image using Canny edges

    :param img: the image we want to get the outline of
    :return: an image representing the outline
    """

    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 300)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges

def checkIfInRange(x, y, range):
    """
    Check if all x and y coordinates are within a certain range

    :param x: a list of x-coordinates
    :param y: a list of y-coordinates
    :param range: the range we want the coordinates to be within [range_min, range_max]
    :return: True if all are within the range, False if not
    """

    XinRange = all([True if (xc >=range[0] and xc<=range[1]) else False for xc in x])
    YinRange = all([True if (yc >=range[0] and yc<=range[1]) else False for yc in y])
    return (XinRange and YinRange)


def create_training_sample(pathToOff, pathToTrainingFolder, idx):
    """
    This function creates a single training data sample based on an .off file

    :param pathToOff: The loacation of the .off file that we want to make a training data sample from
    :param pathToTrainingFolder: The path to the folder where we want to save the value
    :param idx: The index we append to the end of the file when we save the training data sample
    :return: --
    """

    # Augmentation
    scale_range = [0.5, 1.0] # How muc hwe increase or decrease the size of the pointcloud
    coordinates_noise = 5   # how big the size of the noise should be that we randomly add to the coordinates

    # Read the vertices (corners) , faces (planes) and edges (... edges)
    verts_read, faces_read, num_verts, num_faces, num_edges = read_off(pathToOff)

    # Do the resize of the verts
    x, y, z = zip(*verts_read)

    # average around (0,0,0)
    meanx, meany, meanz = np.mean(x), np.mean(y), np.mean(z)
    x = [xc - meanx for xc in x]
    y = [yc - meanx for yc in y]
    z = [zc - meanx for zc in z]

    # rescale
    minlim, maxlim, diff = getLim(x, y, z)  # we want it between -150 and 150

    # Scale so they are of the same size
    scale = 112 / max(abs(minlim), abs(maxlim)) # scale to make sure they are of the same size
    scale_augmentation = scale_range[0] + random.random() * (scale_range[1] - scale_range[0]) # scale to augment data

    x = [xc * scale * scale_augmentation for xc in x]
    y = [yc * scale * scale_augmentation for yc in y]
    z = [zc * scale * scale_augmentation for zc in z]

    # Check if x and y are inside of the range
    if not checkIfInRange(x, y, [-100, 100]):
        while True:
            x = [xc * 0.95 for xc in x]
            y = [yc * 0.95 for yc in y]
            z = [zc * 0.95 for zc in z]
            if checkIfInRange(x, y, [-100, 100]):
                break

    # Add some noise to the coordinates
    x = [xc + random.randint(-coordinates_noise, coordinates_noise) for xc in x]
    y = [yc + random.randint(-coordinates_noise, coordinates_noise) for yc in y]
    z = [zc + random.randint(-coordinates_noise, coordinates_noise) for zc in z]

    # Only create pointclouds from the front
    ang_x, ang_y, ang_z = -90, int(random.random() * 180 - 90), 0
    for i_coor, (xc, yc, zc) in enumerate(zip(x, y, z)):
        point_rotated = rotate_point_3d([xc, yc, zc], math.radians(ang_x), math.radians(ang_y), math.radians(ang_z))
        verts_read[i_coor] = point_rotated

    # testing saving the .off file
    save_OFF_file('temp.off', verts=verts_read, faces=faces_read, num_edges=num_edges)

    # Reading the .off file and turning it into a drawn thing.
    verts_new, faces_new, num_verts, num_faces, num_edges = read_off('temp.off')

    x_new, y_new, z_new = zip(*verts_new)

    # recenter to (112, 112), which is the center of our image
    imgSize = 224
    x_new = [xc + imgSize/2 for xc in x_new]
    y_new = [imgSize - (yc + imgSize/2) for yc in y_new]

    img = np.zeros((imgSize, imgSize, 3))
    for face in faces_new:
        img = drawPolygons(img, x_new, y_new, face[1:], color=(255, 255, 255))

    img_edges = getContour(img)

    # Get a pointcloud
    pcloud = get_pcloud('temp.off', 1024)

    # save both the pointcloud and our image
    np.save(pathToTrainingFolder + f"/sample_{idx}.npy", pcloud)
    cv2.imwrite(pathToTrainingFolder + f"/sample_{idx}.png", img_edges)


if __name__ == "__main__":

    # What we need to input
    pathToTrainingData = 'Path/To/Where/We/Want/To/Save/TrainingData'  # where we want to save our training data
    locationToKerasFiles = f'Path/To/.keras/datasets'                  # path to where we store the keras datasets on this machine
    pathToSamples = 'SampleTrainingData'                               # where we want to save a couple of visualisations of our training data
    samplePerOffFile = 50                                              # how many training samples we make from each OFF file

    # We use the ModelNet10 model dataset, the smaller 10 class version of the ModelNet40 dataset. First download the data:
    DATA_DIR = tf.keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
    )
    DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")
    print(f"We stored the data at : {DATA_DIR}")

    # Load all the paths to our chair images
    foldersWithSample = [f'{locationToKerasFiles}/ModelNet10/chair/train/',
                         f'{locationToKerasFiles}/ModelNet10/chair/test/']
    AllOffPaths = []
    for folder in foldersWithSample:
        AllOffPaths.extend([folder + f"/{fileName}" for fileName in os.listdir(folder) if '.off' in fileName])
    print(f"Total number of OFF files : {len(AllOffPaths)}")

    # Samples per off file
    totTrainingSamples = samplePerOffFile * len(AllOffPaths)
    print(f"Total number of trainingsamples to be made : {totTrainingSamples}")

    # Get the path to the resulting images
    if not os.path.exists(pathToTrainingData): os.mkdir(pathToTrainingData)
    idx = 0

    # Read the cad file (which is in .off format)
    for i_off, offPath in enumerate(AllOffPaths):
        # create the folder we put the images in
        pathToTrainingData_cur = pathToTrainingData + f"/chair_{i_off}"
        if not os.path.exists(pathToTrainingData_cur): os.mkdir(pathToTrainingData_cur)
        [os.remove(pathToTrainingData_cur + f"/{fileName}") for fileName in os.listdir(pathToTrainingData_cur)]

        for _ in range(samplePerOffFile):
            create_training_sample(offPath, pathToTrainingData_cur, idx)
            idx += 1
            if idx % 100 == 0:
                print(f"We made sample number : {idx} / {totTrainingSamples}, we are at : {int(100 *idx/totTrainingSamples)} %")

    # cleanup
    if os.path.exists('temp.off'): os.remove('temp.off')

    # VISUALISATION : Show some of the training data that we created.
    allPathToImg = []
    all_fold = [pathToTrainingData + f"/{fileName}" for fileName in os.listdir(pathToTrainingData)]
    for class_i in range(len(all_fold)):
        fold_cur = all_fold[class_i]
        for imgPath in [fold_cur + f"/{imgPath}" for imgPath in os.listdir(fold_cur) if '.png' in imgPath]:
            allPathToImg.append(imgPath)
    random.shuffle(allPathToImg)

    # Sample training data
    if not os.path.exists(pathToSamples): os.mkdir(pathToSamples)
    for fileName in os.listdir(pathToSamples):
        os.remove(f'{pathToSamples}/{fileName}')

    for i in range(100):
        imgCur = cv2.imread(allPathToImg[i])
        pcloud = np.load(allPathToImg[i][:-4]+'.npy')

        x, y, z = pcloud[:, 0], pcloud[:, 1], pcloud[:, 2]
        imgPcloud = plot_point_cloud(x, y, z, allPathToImg[i][:-4]+'.npy')
        imgPcloud = cv2.resize(imgPcloud, [imgCur.shape[0], imgCur.shape[1]])

        imgTot = np.concatenate([imgCur, imgPcloud], axis=1)

        cv2.imwrite(f'{pathToSamples}/Sample_{i}.png', imgTot)

        # cv2.imshow('Training Data sample', imgTot)
        # cv2.waitKey(-1)

    print(f"We have saved {i} visualisations in {pathToSamples}")
