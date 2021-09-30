'''from PIL import Image

#read the image
im = Image.open("34yre0ai35221.jpg")

#show image
im.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('110ed2b7a4d5c5746eeb71b1b2d84cdd-imagejpeg.jpg')
imgplot = plt.imshow(img, interpolation='nearest')
plt.axis("off")
#plt.savefig('34yre0ai35221.jpg',bbox_inches='tight',pad_inches=0)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()


# import the cv2 library
import cv2

# The function cv2.imread() is used to read an image.
img_grayscale = cv2.imread('110ed2b7a4d5c5746eeb71b1b2d84cdd-imagejpeg.jpg')

# The function cv2.imshow() is used to display an image in a window.
cv2.imshow("color image", img_grayscale)

# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
cv2.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()

import cv2
import ctypes

# Get the window size and calculate the center
user32 = ctypes.windll.user32
win_x, win_y = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)] 
win_cnt_x, win_cnt_y = [user32.GetSystemMetrics(0)/2, user32.GetSystemMetrics(1)/2] 

# load image
imgwindow = 'show my image'
image = cv2.imread('110ed2b7a4d5c5746eeb71b1b2d84cdd-imagejpeg.jpg',cv2.IMREAD_UNCHANGED)

# Get the image size information
off_height, off_width = image.shape[:2]
off_height /= 2
off_width /= 2

# Show image and move it to center location
image = cv2.resize(image,(win_x, win_y))
cv2.imshow(imgwindow,image)

cv2.moveWindow(imgwindow,0,0)
cv2.waitKey(0)
cv2.destroyAllWindows()



import cv2
import ctypes

user32 = ctypes.windll.user32
win_x, win_y = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)] 
win_cnt_x, win_cnt_y = [user32.GetSystemMetrics(0)/2, user32.GetSystemMetrics(1)/2] 

img = cv2.imread('110ed2b7a4d5c5746eeb71b1b2d84cdd-imagejpeg.jpg', cv2.IMREAD_UNCHANGED,(win_x, win_y))
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import ctypes


images = [cv2.imread(file) for file in files]



user32 = ctypes.windll.user32
win_x, win_y = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)] 
win_cnt_x, win_cnt_y = [user32.GetSystemMetrics(0)/2, user32.GetSystemMetrics(1)/2] 



 

 
scale_percent = 180 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img,(win_x,win_y), interpolation = cv2.INTER_AREA)
 

cv2.imshow("Resized image", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''

import cv2
import numpy as np
import glob
import random
import ctypes
import os
import sys
from tkinter import Tk, filedialog

root=Tk()
root.withdraw()
open_file = filedialog.askdirectory() # Returns opened path as str
print(open_file)

user32 = ctypes.windll.user32
win_x, win_y = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)] 
win_cnt_x, win_cnt_y = [user32.GetSystemMetrics(0)/2, user32.GetSystemMetrics(1)/2] 

global qList
qList = []
last_dir = ''
ext = ['png', 'jpg','jpeg','bmp','webp','jfif','tiff',]    # Add image formats here

directories = [x[0].replace('\\','/') for x in os.walk(open_file)]
files = []
for d in directories: 
    [files.extend(glob.glob(d+'/*.' + e)) for e in ext]
print(len(files))
def resizeAndPad(img, size, padColor=0):
    try:
        h, w = img.shape[:2]
        sh, sw = size
    except Exception:
        print('Something is wrong with '+last_dir)

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


'''
h_img = cv2.imread(files[random.randint(0,len(files)-1)]) # horizontal image
scaled_h_img = resizeAndPad(h_img, (2550,1070), 127)

sq_img = cv2.imread(files[random.randint(0,len(files)-1)]) # square image
scaled_sq_img = resizeAndPad(sq_img, (win_y-10,win_y-10), 127)

'''

def getIm():
    global last_dir
    last_dir = files[random.randint(0,len(files)-1)]
    v_img = cv2.imread(last_dir) 
    scaled_v_img = resizeAndPad(v_img, (1070,2400), 127)
    cv2.imshow("SlideShow", scaled_v_img)
    cv2.moveWindow('SlideShow',0,0)
    
getIm()

while(1):
    res = cv2.waitKey(0)
    if res == ord('e'):
        getIm()
    elif res == ord('w'):
        print('lal')
    elif res == ord('m'):
        print(last_dir+' marked for quarantine!')
        qList.append(last_dir)
    elif res == ord('z'):
        file = open('qList',mode='a')
        for x in qList:
            file.write(x)
        file.close()
        print(str(len(qList))+' qlist item. exiting program')
        cv2.destroyAllWindows()
        sys.exit(0)


