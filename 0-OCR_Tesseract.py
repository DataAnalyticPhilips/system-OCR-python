from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit, QDialog
from PyQt5 import uic, QtCore, QtGui, QtWidgets, QtSerialPort
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget # Show-GUI
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap # Images
from PyQt5.QtCore import QTimer, QDateTime, Qt, pyqtSlot  # Timer !
from datetime import datetime
from imutils.object_detection import non_max_suppression

import cv2, time, imutils, pytesseract
import sys, requests, json, re, uuid, numpy as np

#Filtering Image
kernel = np.ones((2,2),np.uint8)

#Counter Image got Captured
img_counter = 0
img_counter_tesseract = 0

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("OCR_Design.ui", self)
        
        # Button
        self.PushButton_Open_Camera.clicked.connect(self.OpenCamera)
        self.PushButton_Close_Camera.clicked.connect(self.CloseCamera)
        self.PushButton_Close_Camera.hide()
        self.PushButton_Capture_Image.clicked.connect(self.CaptureImage)
        #self.PushButton_Save.clicked.connect(self.SaveImage)
    
        #Set timer timeout callback function
        self.timer = QTimer()
        self.timer.timeout.connect(self.ImageDetect)
        
        #Set Capture
        self.logic=0
        self.value=1
        
    def CaptureImage(self): 
        self.logic=2
        
    def OpenCamera(self):
        self.PushButton_Open_Camera.hide()
        self.PushButton_Close_Camera.show()
        print('OpenCamera')
        if not self.timer.isActive():
            self.cap = cv2.VideoCapture(0)
            self.timer.start(20)
            
        else:
            self.timer.stop()
            self.cap.release()
    
    def ImageDetect(self):
        global image
        ret, image = self.cap.read()
        image = imutils.resize(image, width=320)
        image = imutils.resize(image, height=240)
        self.ShowImage(image,1)
        
    def ShowImage(self, img, window=1):
        global img_counter
        qformat = QImage.Format_Indexed8
        img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
        img = img.rgbSwapped()
        self.Picture_1.setPixmap(QPixmap.fromImage(img))
        self.Picture_1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        
        if(self.logic==2):
            print("Captured")
            img_name = "/home/pi/Documents/00.Github/Image/Capture/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, image)
            print("{} written!".format(img_name))
            img_counter += 1
            print(img_counter)
            self.logic=1
            self.Screenshot()
        
        else:
            print('Return Not Found')
        
    def CloseCamera(self):
        print('CloseCamera')
        self.PushButton_Close_Camera.hide()
        self.PushButton_Open_Camera.show()
        self.timer.stop()
        self.cap = cv2.destroyAllWindows()
        self.Picture_1.clear()
        self.Picture_2.clear()
        
    def Screenshot(self):
        input_image = cv2.imread('/home/pi/Documents/00.Github/Image/Capture/opencv_frame_{}.png'.format(img_counter-1))
        # convert image to RGB format
        image_screenshot = imutils.resize(input_image, width=320)
        image_screenshot = imutils.resize(input_image, height=240)
        
        qformat = QImage.Format_Indexed8
        img = QImage(image_screenshot, image_screenshot.shape[1], image_screenshot.shape[0], QImage.Format_RGB888)
        img_screenshot = img.rgbSwapped()
        print('Ocr a East')
        self.OCRaEAST()

#Tesseract Mode -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def OCRaEAST(scores, geometry):
            # grab the number of rows and columns from the scores volume, then
            # initialize our set of bounding box rectangles and corresponding
            # confidence scores
            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []
            # loop over the number of rows
            for y in range(0, numRows):
                # extract the scores (probabilities), followed by the
                # geometrical data used to derive potential bounding box
                # coordinates that surround text
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]
                # loop over the number of columns
                for x in range(0, numCols):
                    # if our score does not have sufficient probability,
                    # ignore it
                    if scoresData[x] < 0.5:
                        continue
                    # compute the offset factor as our resulting feature
                    # maps will be 4x smaller than the input image
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)
                    # extract the rotation angle for the prediction and
                    # then compute the sin and cosine
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)
                    # use the geometry volume to derive the width and height
                    # of the bounding box
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]
                    # compute both the starting and ending (x, y)-coordinates
                    # for the text prediction bounding box
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)
                    # add the bounding box coordinates and probability score
                    # to our respective lists
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])
            # return a tuple of the bounding boxes and associated confidences
            return (rects, confidences)

    def OCRaEAST(self):
        print('OCRaEAST')
        global img_counter_tesseract
        global img_counter_east_cropped
        # load the input image and grab the image dimensions
        image = cv2.imread('/home/pi/Documents/00.Github/Image/Capture/opencv_frame_{}.png'.format(img_counter-1))
        orig = image.copy()
        (origH, origW) = image.shape[:2]
        
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (320, 320)
        rW = origW / float(newW)
        rH = origH / float(newH)
        
        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested in -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        
        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet("/home/pi/Documents/1.Program/03-04-2022/frozen_east_text_detection.pb")

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < 0.5:
                    continue
                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        # return a tuple of the bounding boxes and associated confidences
    
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        
        # initialize the list of results
        results = []
        
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            global text
            # scale the bounding box coordinates based on the respective ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            
            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * 0.0065)
            dY = int((endY - startY) * 0.168)
            
            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))
            
            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]
            
            print('startX : ',startX)
            print('startY : ',startY)
            print('endX   : ',endX)
            print('endY   : ',endY)
            cropimage = orig[startY:endY,startX:endX]
            #cv2.imshow("CropImage", cropimage)
            
            # convert EAST and cropping to show in GUI
            cropping_image = cropimage.copy()
            image_croppings = imutils.resize(cropping_image, width=320)
            image_croppings = imutils.resize(cropping_image, height=240)
            
            #Filtering in GUI
            gray = cv2.cvtColor(cropimage, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("Text Gray", gray)

            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            #cv2.imshow("Text GaussianBlur", gray)

            thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)[1]
            #cv2.imshow("Text THRESH_BINARY", thresh)

            thresh = cv2.dilate(thresh,kernel,iterations = 1)
            #cv2.imshow("Text Dilate", thresh)
    
            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 1, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(thresh, config=config)
            # add the bounding box coordinates and OCR'd text to the list
            # of results
            results.append(((startX, startY, endX, endY), text))
            
        # sort the results bounding box coordinates from top to bottom
        results = sorted(results, key=lambda r:r[0][1])
        
            # loop over the results
        for ((startX, startY, endX, endY), text) in results:
            # display the text OCR'd by Tesseract
            print("OCR TEXT")
            print("========")
            print("{}\n".format(text))
            print("Text :  ",text.strip())
            
            #Set Text Label
            self.Output_Text1.setText(text) #Set Text Output Label 1
            
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw the text and a bounding box surrounding
            # the text region of the input image
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            output = orig.copy()
            cv2.rectangle(output, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # show the output image
            # convert image to RGB format
            image_tesseract = imutils.resize(output, width=320)
            image_tesseract = imutils.resize(output, height=240)
            
            # save image tesseract
            img_tesseract_results = "/home/pi/Documents/00.Github/Image/Results_OCR/opencv_tesseract_frame_{}.png".format(img_counter_tesseract)
            cv2.imwrite(img_tesseract_results, image_tesseract)
            print("{} written!".format(img_tesseract_results))
            img_counter_tesseract += 1
            print(img_counter_tesseract)
            
            qformat = QImage.Format_Indexed8
            img = QImage(output, image_tesseract.shape[1], image_tesseract.shape[0], QImage.Format_RGB888)
            image_tesseract = img.rgbSwapped()
            self.Picture_2.setPixmap(QPixmap.fromImage(image_tesseract))
            self.Picture_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            print('Tesseract')
            cv2.waitKey(0)
#Tesseract Mode -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    #window.startTimer()
    sys.exit(app.exec())
    