import cv2
import numpy as np

def change(x):
    pass

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)                           #matrix of zeros with dimension of img
    match_mask_color = 255                              #this results in a tuple(255,255,255)
    cv2.fillPoly(mask, vertices, match_mask_color)      #this results in a white polygon (triangle in this case)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):

    #to make the lines look a bit transparent, draw them on a blank img
    # and then use add/addWeighted
    blank = np.zeros(img.shape, dtype=np.uint8)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(blank, (x1,y1),(x2,y2),(0,255,0),3)

    img = cv2.addWeighted(img, 0.8, blank, 1, 0)

    return img

def process(img):
    height = img.shape[0]
    width = img.shape[1]

    ROI_vertices = [(0,0.95*height),(0.5*width,0.75*height),(width,0.95*height)]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    canny = cv2.Canny(gray, 100, 150)

    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)

    cropped_img = region_of_interest(dilate, np.array([ROI_vertices],np.int32))
    #here np.array() makes the same array as ROI_vertices with
    # datatype of 32 bit int

    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 50,
                           minLineLength=20, maxLineGap=40)
    #try adjusting these 4 parameters if some lines are not detected

    if lines is None:
        img_with_lines = img
    else:
        img_with_lines = draw_lines(img, lines)
    return img_with_lines

cap = cv2.VideoCapture('lane_test2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture('lane_test2.mp4')
        continue
    cv2.imshow('orig',frame)
    frame = process(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


