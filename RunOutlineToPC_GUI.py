import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import cv2
import io
import onnxruntime
import time
import os

l = [0, 0]      # global array to store the images we display in

"""
    This is the main function that runs the GUI that allows us to draw an outline of a chair, and receive a pointcloud
    after running a model.
    
    The trainingdata on which the model is trained is created from the publicly available dataset of 3d objects, 
    represented by .off files. More information about this dataset can be found at 
    https://3dvision.princeton.edu/projects/2014/3DShapeNets/, but we download the data using tf.keras.utils.get_file. 
    More information about how we turn the .off files to actual training data can be found in CreateTrainingData.py
    
    
    Author : Martijn Folmer
    Date : 12-08-2023

"""

class DrawingApp:
    def __init__(self, window):

        # Bools that make the gui function
        self.found_pc = False  # if this is true, we have found a pointcloud to display
        self.DoShameLessSelfPromotion = True  # if set to true, we will draw my name (martijn folmer) in bottom right corner. Just set to false ;-)

        # TKINTER variables
        self.window = window
        self.canvasSize = (500, 500)
        self.canvasImg = np.zeros((self.canvasSize[0], self.canvasSize[1], 3))   # the background

        # Create the frames which contain all the tkinter elements
        self.canvasFrame = self.initializeCanvasFrame()
        self.buttonFrame = self.initializeButtonFrame()
        self.horizontalFrame = self.initializeImageFrame()
        self.verticalFrame = self.initializeImageFrame()

        # the original Pointcloud frame
        self.horizontalPCframe_or = np.ones((self.canvasSize[0], self.canvasSize[1], 3))*255
        self.verticalPCframe_or = np.ones((self.canvasSize[0], self.canvasSize[1], 3))*255

        # Create the labels we can get images/ pointclouds to
        self.horizontalLabel = self.createImgLabel(self.horizontalFrame)
        self.verticalLabel = self.createImgLabel(self.verticalFrame)

        # Testing, load images and create the labels we store with them
        self.loadImage(self.horizontalLabel, self.horizontalPCframe_or, 0)
        self.verticalPCframe_or = self.ShamelessSelfPromotion(self.verticalPCframe_or)
        self.loadImage(self.verticalLabel, self.verticalPCframe_or, 1)

        # Bind the buttons to the canvas functions
        self.canvasFrame.bind("<Button-1>", self.start_drawing)
        self.canvasFrame.bind("<Button-3>", self.start_drawing)
        self.canvasFrame.bind("<B1-Motion>", self.draw_white)
        self.canvasFrame.bind("<B3-Motion>", self.draw_black)

        # ONNX models
        pathToOnnxModel = 'ModelsOutlineToPC/OutLineToPC_chair_1024_all.onnx'
        self.session, self.inputName, self.inputShape = self.initializeONNX(pathToOnnxModel)

        # The output pointclouds
        self.x_or, self.y_or, self.z_or = [], [], []

        # Rotations
        self.rotation_horizontal = [0, 0, 0]
        self.rotation_speed_horizontal = [0, 0.1, 0]
        self.rotation_vertical = [0, 0, 0]
        self.rotation_speed_vertical = [-0.1, 0, 0]

        # Save npy
        self.SaveNamePCFold = "ExportedPointclouds/"  # Where we want to export our pointclouds to
        self.SaveNamePC = "CreatedPointcloud"  # the base name of how we save the numpy file with the pointcloud


    # Functions for initializing the TKINTER parts
    def initializeImageFrame(self):
        frame_img = tk.Frame(master=self.window, borderwidth=5)
        frame_img.pack(side = tk.LEFT)
        return frame_img

    def initializeCanvasFrame(self):
        canvas = tk.Canvas(master=self.window, bg="black", borderwidth=5)
        canvas.config(width=self.canvasSize[0], height = self.canvasSize[1])
        canvas.pack(side=tk.LEFT)
        return canvas

    def initializeButtonFrame(self):
        frame_buttons = tk.Frame(master=self.window, borderwidth=5)
        frame_buttons.pack(side=tk.LEFT)

        b1 = tk.Button(
            frame_buttons,
            text="Run Model",
            command=self.runModel
        )
        b1.pack(side=tk.TOP)

        b2 = tk.Button(
            frame_buttons,
            text="Clear Canvas",
            command=self.clearCanvas
        )
        b2.pack(side=tk.TOP)
        b3 = tk.Button(
            frame_buttons,
            text="Export pointcloud",
            command=self.exportPointCloud
        )
        b3.pack(side=tk.TOP)

        return frame_buttons

    def createImgLabel(self, _frame):
        # Create the label with the image
        label = tk.Label(master=_frame)
        label.pack()
        return label


    def clearCanvas(self):
        """
        This clears the space where we draw the outline, as well as the pointclouds (if we have any) and the image
        that we use to copy from the canvas (which is the image we pass on to the ML model)

        :return: --
        """

        # Clear the canvas and canvas Img
        self.canvasFrame.delete("all")
        self.canvasImg = np.zeros((self.canvasSize[0], self.canvasSize[1], 3))   # the background

        # reset the pointcloud images
        self.loadImage(self.horizontalLabel, self.horizontalPCframe_or, 0)
        self.verticalPCframe_or = self.ShamelessSelfPromotion(self.verticalPCframe_or)
        self.loadImage(self.verticalLabel, self.verticalPCframe_or, 1)
        self.found_pc = False       # reset whether we have found a pointcloud

    def loadImage(self, _label, _imgToAdd, _idx):
        """
        turn pointcloud image to a tkinter image that we can display

        :param _label:      the label of the element in our GUI window that we want to upload the image onto
        :param _imgToAdd:   The image we want to show up
        :param _idx:        The index where we store the image in our global array, so the images actually show up
        :return: --
        """
        #

        # convert image to the Tkinter image
        img = np.asarray(_imgToAdd, dtype=np.uint8)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)

        # append to global array, so it stays in memory
        l[_idx] = img_tk

        # Create the label with the image
        _label.config(image=img_tk)

    #########################
    # Drawing on the canvas
    #########################
    def start_drawing(self, event):
        """
        When we press the button down for the first time when we start drawing
        """

        self.last_x = event.x
        self.last_y = event.y

    def draw(self, color, event, _width):
        """
        When we hold down the button to draw the lines
        """

        if self.last_x and self.last_y:
            x, y = event.x, event.y
            self.canvasFrame.create_line(self.last_x, self.last_y, x, y, fill=color, width=_width)
            color_RGB = (255, 255, 255) if color == "white" else (0, 0, 0)
            self.canvasImg = cv2.line(self.canvasImg, (int(self.last_x), int(self.last_y)), (int(x), int(y)),
                                      color_RGB, _width)
            self.last_x, self.last_y = x, y

    def draw_white(self, event):
        """ Drawing a white line """
        self.draw("white", event, 6)

    def draw_black(self, event):
        """ Drawing a black line """
        self.draw("black", event, 8)

    #################
    # Pointclouds and the rotation of points
    #################

    def exportPointCloud(self):
        """
        When pressing the "Export Pointcloud" button, we save the pointcloud which has been output by our ML model
        into a numpy file and save it.

        :return: --
        """

        # only export the pointcloud if we have something to save (so if we have found a pointcloud)
        if self.found_pc:
            # Create the parent folder where we save the exported pointclouds if it does not already exist
            if not os.path.exists(self.SaveNamePCFold): os.mkdir(self.SaveNamePCFold)

            # Find a name to save the pointcloud under by incrementing kn and finding a name that hasn't been taken
            # ...Yes, this is an inefficient method. No, I'm not going to change it.
            kn = 0
            while True:
                saveNameCur = self.SaveNamePCFold + self.SaveNamePC + f"{kn}.npy"
                if not os.path.exists(self.SaveNamePCFold + saveNameCur):
                    break

            # Save the pointcloud in the pointcloud folder
            coordinatesToSave = [[xc, yc, zc] for (xc, yc, zc) in zip(self.x_or, self.y_or, self.z_or)]
            np.save(saveNameCur, np.asarray(coordinatesToSave))
            print(f"We exported a pointcloud to : {saveNameCur}")

    def drawPointCloud(self, x, y):
        """
        Draw a 2D representation of the pointcloud, by drawing its x and y coordinates
        
        :param x: an array of x coordinates that we want to draw
        :param y: an array of y coordinates that we want to draw
        :return:
        """

        img = np.ones((self.canvasSize[0], self.canvasSize[1], 3))*255
        for (xc, yc) in zip(x, y):
            img = cv2.circle(img, (int(xc + self.canvasSize[0]/2), int(self.canvasSize[1] - (yc + self.canvasSize[1]/2))), 3, (0, 0, 0), 1)

        return img

    def rotation_matrix_x(self, theta_x):
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

    def rotation_matrix_y(self, theta_y):
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

    def rotation_matrix_z(self, theta_z):
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

    def rotate_point_3d(self, point, theta_x, theta_y, theta_z):
        """
        Rotate a 3D point around the global axes (x, y, z) by angles (theta_x, theta_y, theta_z).

        :param point: 1D numpy array representing the 3D point [x, y, z].
        :param theta_x: Rotation angle around the x-axis in radians.
        :param theta_y: Rotation angle around the y-axis in radians.
        :param theta_z: Rotation angle around the z-axis in radians.
        :return: The rotated 3D point as a 1D numpy array.
        """
        R_x = self.rotation_matrix_x(theta_x)
        R_y = self.rotation_matrix_y(theta_y)
        R_z = self.rotation_matrix_z(theta_z)

        # Combine the rotation matrices (order matters: first X, then Y, and finally Z)
        R_combined = np.dot(R_z, np.dot(R_y, R_x))

        # Rotate the point by multiplying with the combined rotation matrix
        rotated_point = np.dot(R_combined, point)

        return rotated_point

    def rotateAllPoints(self, x_or, y_or, z_or, ang_x, ang_y, ang_z):
        """
        When we want to rotate all points by a certain set of angles

        :param x_or: list of x- coordinates
        :param y_or: list of y-coordinates
        :param z_or: list of z-coordinates
        :param ang_x: angle around x-axis that we rotate
        :param ang_y: angle around y-axis that we rotate
        :param ang_z: angle around z-axis that we rotate
        :return: 3 list with the newly rotated x,y and z coordinates.
        """

        allPoints = [[xc, yc, zc] for (xc, yc, zc) in zip(x_or, y_or, z_or)]
        allPoints = [self.rotate_point_3d(point, ang_x, ang_y, ang_z) for point in allPoints]
        x, y, z = zip(*allPoints)
        return x, y, z

    def MakeSureRotationIsBetweenLim(self, rot):
        """check if a rotation [ang_x, ang_y, ang_z] has its angles between 0 and 360"""
        for i in range(3):
            if rot[i] < 0: rot[i] += 360
            if rot[i] > 360: rot[i] -= 360

        return rot

    def AddSpeedToRotation(self, rot, rot_v):
        """ The pointclouds rotate around a certain axis when displayed. This function just increments angles with a
        certain angular speed """
        rot = [(rotation + speed) for (rotation, speed) in zip(rot, rot_v)]
        rot = self.MakeSureRotationIsBetweenLim(rot)
        return rot

    ##################
    # Run the ONNX Model to turn an outline to a pointcloud
    ##################

    def initializeONNX(self, _PathToONNX):
        """
        Initialize the .onnx machine learning model and return everything that we need to run it.

        :param _PathToONNX: The file path to the where we stored the .onnx model
        :return: session (what we need to invoke), inputName (under what name we set the input), inputShape (how
        we need to reshape our input in order to run it)
        """

        session = onnxruntime.InferenceSession(_PathToONNX, providers=['CPUExecutionProvider'])
        inputName = session.get_inputs()[0].name
        inputShape = session.get_inputs()[0].shape
        inputShape = [1, inputShape[1], inputShape[2], inputShape[3]]
        return session, inputName, inputShape

    def RunONNXmodel(self, _inputImg, _session, _inputName, _inputShape):
        """
        Invoking a single instance of our model.

        :param _inputImg: the input we want to run in the .onnx model
        :param _session:  The session which represents the initialized model
        :param _inputName: The name under which we set the input
        :param _inputShape: The shape of the input required to run the model

        :return: the output from the model
        """

        _inputImg = np.reshape(_inputImg, _inputShape)
        _inputImg = _inputImg.astype(dtype=np.float32)
        outputs = _session.run(None, {_inputName: _inputImg})  # run session and return it
        return outputs[0]


    def runModel(self):
        """
        Run our .onnx model. Includes getting the image from our canvas, preprocessing it, running it and postprocessing
        it. The output is a pointcloud which we display.

        :return: --
        """
        # Get the canvas image
        inputImg = np.asarray(self.canvasImg, dtype=np.uint8)
        scaling = inputImg.shape[0]/self.inputShape[1]
        inputImg = cv2.resize(inputImg, (self.inputShape[1], self.inputShape[2]))

        # Run the model
        pointcloud = self.RunONNXmodel(inputImg, self.session, self.inputName, self.inputShape)

        # subdivide into x_or, y_or, z_or
        self.x_or = pointcloud[0, :, 0]
        self.y_or = pointcloud[0, :, 1]
        self.z_or = pointcloud[0, :, 2]

        # Expand, so it looks a little more like the outline canvas
        self.x_or = [coor * scaling for coor in self.x_or]
        self.y_or = [coor * scaling for coor in self.y_or]
        self.z_or = [coor * scaling for coor in self.z_or]

        # Draw pointcloud and add to image frames
        img_horizontal = self.drawPointCloud(self.x_or, self.y_or)
        self.loadImage(self.horizontalLabel, img_horizontal, 0)

        img_vertical = self.drawPointCloud(self.x_or, self.y_or)
        img_vertical = self.ShamelessSelfPromotion(img_vertical)
        self.loadImage(self.verticalLabel, img_vertical, 1)
        self.found_pc = True

    def updatePC(self):
        """
        We want the pointclouds to rotate, so this function does just that and alters the pointclouds based on an
        angle which gets updated every frame, then displays it.

        :return: --
        """


        # update the pointclouds with our keypoints
        if self.found_pc:
            hor_x, hor_y, _ = self.rotateAllPoints(self.x_or, self.y_or, self.z_or, self.rotation_horizontal[0], self.rotation_horizontal[1], self.rotation_horizontal[2])
            img_horizontal = self.drawPointCloud(hor_x, hor_y)
            self.loadImage(self.horizontalLabel, img_horizontal, 0)

            ver_x, ver_y, _ = self.rotateAllPoints(self.x_or, self.y_or, self.z_or, self.rotation_vertical[0], self.rotation_vertical[1], self.rotation_vertical[2])
            img_vertical = self.drawPointCloud(ver_x, ver_y)
            img_vertical = self.ShamelessSelfPromotion(img_vertical)
            self.loadImage(self.verticalLabel, img_vertical, 1)

    def ShamelessSelfPromotion(self, img):
        """
         Shameless self promotion, draw my name in the bottom right. We do this just to make it slightly easier for me
         to show off who made this in videos, but it has no bearing on functionality. Just set
         self.DoShameLessSelfPromotion to false if you don't want to display the name ;-)

         ... Though I would appreciate a like, or a star, or a kudos, or whatever it is wherever you found this
         script. Have a nice day!
        """
        if self.DoShameLessSelfPromotion:
            return cv2.putText(img, "Martijn Folmer", (self.canvasSize[0] - 240, self.canvasSize[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            return img


if __name__ == "__main__":
    time_cur = 0.01        # We use this as a time step for adding rotational speed to the rotations of the pointclouds
    time_c = time.time()

    # initialize the Tkinter window.
    window = tk.Tk()
    DA = DrawingApp(window)

    while True:
        if time.time()-time_c >= time_cur:
            # Update the rotations
            DA.rotation_horizontal = DA.AddSpeedToRotation(DA.rotation_horizontal, DA.rotation_speed_horizontal)
            DA.rotation_vertical = DA.AddSpeedToRotation(DA.rotation_vertical, DA.rotation_speed_vertical)
            DA.updatePC()           # update the pointclouds we draw
            time_c = time.time()

        # UPDATE TKINTER WINDOW : equivalent of window.mainloop()
        window.update_idletasks()
        window.update()
