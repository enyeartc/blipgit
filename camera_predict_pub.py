import cv2
cam = cv2.VideoCapture(0)
import time
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import requests

def display_symb():
    # the IoT Device I created can be communicated with via electric imp's website
    # the name XXXXXX1 would be replaced with actual
    r = requests.get('https://agent.electricimp.com/XXXXXX1?image_Location=6&amp;display_Image')
    r = requests.get('https://agent.electricimp.com/XXXXXX2?image_Location=7&amp;display_Image')
    r = requests.get('https://agent.electricimp.com/XXXXXX3?image_Location=8&amp;display_Image')
def display_order(mode):
    if(mode == 0):
        r = requests.get('https://agent.electricimp.com/XXXXXX1?image_Location=1&amp;display_Image')#green turtle
        r = requests.get('https://agent.electricimp.com/rXXXXXX2?image_Location=2&amp;display_Image')#black ying
        r = requests.get('https://agent.electricimp.com/XXXXXX3?image_Location=3&amp;display_Image')#blue house
    elif (mode == 1):
        r = requests.get('https://agent.electricimp.com/XXXXXX1?image_Location=1&amp;display_Image')#green turtle
        r = requests.get('https://agent.electricimp.com/rXXXXXX2?image_Location=3&amp;display_Image')#black ying
        r = requests.get('https://agent.electricimp.com/XXXXXX3?image_Location=2&amp;display_Image')#blue house
    elif (mode == 2):
        r = requests.get('https://agent.electricimp.com/XXXXXX1?image_Location=2&amp;display_Image')#green turtle
        r = requests.get('https://agent.electricimp.com/rXXXXXX2?image_Location=3&amp;display_Image')#black ying
        r = requests.get('https://agent.electricimp.com/XXXXXX3?image_Location=1&amp;display_Image')#blue house
    elif (mode == 3):
        r = requests.get('https://agent.electricimp.com/XXXXXX1?image_Location=3&amp;display_Image')#green turtle
        r = requests.get('https://agent.electricimp.com/rXXXXXX2?image_Location=2&amp;display_Image')#black ying
        r = requests.get('https://agent.electricimp.com/XXXXXX3?image_Location=1&amp;display_Image')#blue house
    elif (mode == 4):
        r = requests.get('https://agent.electricimp.com/XXXXXX1?image_Location=3&amp;display_Image')#green turtle
        r = requests.get('https://agent.electricimp.com/rXXXXXX2?image_Location=1&amp;display_Image')#black ying
        r = requests.get('https://agent.electricimp.com/XXXXXX3?image_Location=2&amp;display_Image')#blue house
    elif (mode == 5):
        r = requests.get('https://agent.electricimp.com/XXXXXX1?image_Location=2&amp;display_Image')#green turtle
        r = requests.get('https://agent.electricimp.com/rXXXXXX2?image_Location=1&amp;display_Image')#black ying
        r = requests.get('https://agent.electricimp.com/XXXXXX3?image_Location=3&amp;display_Image')#blue house

cv2.namedWindow("test")
filename = 'finalized_model.sav'
print('...Load Model')
model = pickle.load(open(filename, 'rb'))

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        display_symb()
        time.sleep(5)
        ret, frame = cam.read()
        if not ret:
            break
        k = cv2.waitKey(1)
        img_rows, img_cols = 100, 200
        # SPACE pressed
        # process the image the same way the training data was created

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray_image, (200, 100))
        images = []
        images.append(small)
        imagesnp = np.array(images)

        if K.image_dim_ordering() == 'th':
            print("== 'th'")
            imagesnp = imagesnp.reshape(imagesnp.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            print("!= 'th'")
            imagesnp = imagesnp.reshape(imagesnp.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        imagesnp = imagesnp.astype('float32')
        imagesnp /= 255

        print('imagesnp shape:', imagesnp.shape)
        print(imagesnp.shape[0], 'train samples')



        prediction = model.predict(imagesnp)
        print(prediction.round(2),np.argmax(prediction))
        display_order(np.argmax(prediction))


cam.release()

cv2.destroyAllWindows()
