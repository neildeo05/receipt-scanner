#!/usr/local/bin/python3.7
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from PIL import Image
import pytesseract
from pytesseract import Output
import sys

FILENAME = sys.argv[1]
OUTPUT_FILENAME = sys.argv[2]
original = cv2.imread(FILENAME)


class Scanner:
    def __init__(self, filename):
        self.filename = filename
        self.img = cv2.imread(self.filename)
        self.small = 500 / self.img.shape[0]
        self.img = cv2.resize(self.img, 
                (
                    int(self.img.shape[1] * self.small),
                    int(self.img.shape[0] * self.small)
                ))
        self.l1 = cv2.GaussianBlur
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        self.l2 = cv2.dilate
        self.l3 = cv2.Canny
        
    def forward(self):
        grayscale = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        x = self.l1(grayscale, (5,5), 0)
        x = self.l2(x, self.kernel)
        x = self.l3(x, 100, 200, apertureSize=3)
        contours, h = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        big_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        smooth_contour = None
        for i in big_contours:
            approx = cv2.approxPolyDP(i, 0.01 * cv2.arcLength(i, True), True)
            if(len(approx) == 4):
                smooth_contour = approx
                break
        assert(type(smooth_contour) != type(None))
        return smooth_contour

    def wrap(self, smooth_contour, img):
        pts = smooth_contour.reshape(4,2)
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        rect /= self.small

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


    @staticmethod
    def plot_image(img, mode="norm"):
        if mode == 'gray':
            plt.figure(figsize=(16,10))
            return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap= 'gray')
        elif mode == 'rgb':
            plt.figure(figsize=(16,10))
            return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.figure(figsize=(16,10))
            return plt.imshow(img)

def main():
    scan = Scanner(FILENAME)
    Scanner.plot_image(original, 'rgb')
    gray = cv2.cvtColor(scan.wrap(scan.forward(), original), cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    result = (gray > T).astype("uint8") * 255
    output = Image.fromarray(result)
    output.save(OUTPUT_FILENAME)
    # Scanner.plot_image(np.array(result), mode='gray')
    # Scanner.plot_image(result, 'gray')


if __name__ == '__main__':
    main()
    plt.show()
