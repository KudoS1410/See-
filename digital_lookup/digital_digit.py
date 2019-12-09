import cv2 as cv
from scipy import ndimage
import numpy as np
import os

def get_images():
    images = []
    folder = os.getcwd()
    for file in os.listdir(folder):
        file_name, file_ext = os.path.splitext(file)
        if file_ext == '.jpg' and file_name != 'img7':
            image = cv.imread(file)
            images.append(image)
    return images

def read_digit(roi, DIGITS_LOOKUP):
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    w = roiW
    h = roiH

    segments = [
        ((dW, 0), (w - dW, dH)),  # top
        ((0, dH), (dW, (h // 2) - dHC)),  # top-left
        ((w - dW, dH), (w, (h // 2) - dHC)),  # top-right
        ((dW, (h // 2) - dHC), (w - dW, (h // 2) + dHC)),  # center
        ((0, (h // 2) + dHC), (dW, h - dH)),  # bottom-left
        ((w - dW, (h // 2) + dHC), (w, h - dH)),  # bottom-right
        ((dW, h - dH), (w - dW, h))  # bottom
    ]
    on = [0] * len(segments)

    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) <= 0.6:
            # cv.rectangle(image, (xA, yA), (xB, yB), color=255, thickness=-1)
            on[i] = 1

    digit = DIGITS_LOOKUP[tuple(on)]
    return(digit)

def see_number(img):
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    image = cv.resize(img, (500, 500))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 142.0, 255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)
    # ret, thresh = cv.threshold(thresh, 10, 255, type=cv.THRESH_BINARY)
    thresh = cv.blur(thresh, (7, 7))
    ret, thresh = cv.threshold(thresh, 250, 255, type=cv.THRESH_BINARY)
    thresh = cv.erode(thresh, (5, 5), iterations=15)
    roi = thresh[185:-70, 200:290].copy()
    roi2 = thresh[185:-70, 295:375].copy()
    roi2 = ndimage.rotate(roi2.copy(), 2.0, reshape=False)
    roi2 = roi2[:, :-15]
    roi3 = thresh[185:-70, 395:490].copy()
    roi3 = ndimage.rotate(roi3.copy(), 3.0, reshape=False)
    roi3 = roi3[:, :-15]
    digit_img = [roi, roi2, roi3]
    digits = []
    for clip in digit_img:
        digits.append(read_digit(clip.copy(), DIGITS_LOOKUP))
    return (digits[0]*10 + digits[1] + digits[2]*0.1)





if __name__ == '__main__':
    images = get_images()
    for image in images:
        number = see_number(image)
        print(number)