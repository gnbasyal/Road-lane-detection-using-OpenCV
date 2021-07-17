import matplotlib.pyplot as plt
import cv2
import numpy as np

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)                           #matrix of zeros with dimension of img
    #channel_count = img.shape[2]
    match_mask_color = 255                              #this results in a tuple(255,255,255)
    cv2.fillPoly(mask, vertices, match_mask_color)      #this results in a white polygon (triangle in this case)
    #cv2.imshow('mask',mask)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):

    #to make the lines look a bit transparent, draw them on a blank img
    # and then use add/addWeighted
    blank = np.zeros(img.shape, dtype=np.uint8)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(blank, (x1,y1),(x2,y2),(0,255,0),10)

    #img = cv2.addWeighted(img, 0.8, blank, 1, 0)
    img = cv2.add(img, blank)
    return img

img = cv2.imread('road.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
height = img.shape[0]
width = img.shape[1]

ROI_vertices = [(0,height),(width/2,height/2),(width,height)]
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 100, 200)
#try adjusting the threshold values if edges are not properly detected
#cv2.imshow('canny',canny)

cropped_img = region_of_interest(canny, np.array([ROI_vertices],np.int32))
#here np.array() makes the same array as ROI_vertices with
# datatype of 32 bit int

#cv2.imshow('cropped',cropped_img)

lines= cv2.HoughLinesP(cropped_img, 6, np.pi/180,160,
                       minLineLength=40,maxLineGap=50)
#try adjusting these 4 parameters if some lines are not detected

#print(lines)
img_with_lines = draw_lines(img, lines)

plt.imshow(img_with_lines)
plt.show()




