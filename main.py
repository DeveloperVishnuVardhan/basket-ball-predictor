import cv2
from colorFinder import ColorFinder
import utils
import numpy as np
import math
import cvzone

# Start with opening the camera.
capture = cv2.VideoCapture('Files/Videos/vid (4).mp4')
myColorFinder = ColorFinder(False)
#hsvVals = {'hmin': 8, 'smin': 124, 'vmin': 13, 'hmax':24, 'smax': 255, 'vmax': 255}
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}
position_listX = []
position_listY = []
x_list = [val for val in range(0, 1300)]

# loop through and display the frames in a window.
while True:
    # Grab the Image.
    success, img = capture.read()
    if not success:
        print("End of Video")
        break

    img = img[0:900, :] # Crop the bottom region.

    # Find the color ball.
    myColorFinder.update(img, hsvVals)
    imgColor, mask = myColorFinder.update(img, hsvVals)
    imgContours, countours = utils.findContours(img, mask, 500)

    if countours:
        position_listX.append(countours[0]['center'][0])
        position_listY.append(countours[0]['center'][1])

    # Find learnable parametres in polynomial Regression.
    if position_listX:
        A, B, C = np.polyfit(position_listX, position_listY, 2)

        for i, (positionX, positionY) in enumerate(zip(position_listX, position_listY)): # Drawing the path.
            cv2.circle(imgContours, (positionX, positionY), 7, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, (positionX, positionY), (positionX, positionY), (0, 255, 0), 2)
            else:
                cv2.line(imgContours, (positionX, positionY), (position_listX[i - 1], position_listY[i - 1]), (0, 255, 0), 2)

        for x in x_list:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        # Perform prediction only up to first 10-frames.
        if len(position_listX) < 10:
            # Prediction
            # X values 330 to 430, Y 590.
            a = A
            b = B
            c = C - 590

            x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 430

        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 150), scale=5,
                               thickness=5, colorR=(0,200,0),offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 150), scale=5, 
            thickness=5, colorR=(0, 0, 200), offset=20)





    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7) # Resize the frame.
    cv2.imshow("Thrsholding", imgContours)
    k = cv2.waitKey(100)
