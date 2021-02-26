import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
img = cv2.imread(filename)
d = pytesseract.image_to_data(img, output_type=Output.DICT)
for i in range(len(d['level'])):
    (x,y,w,h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    img = cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0, 0), 2)


plt.imshow(img);

print(pytesseract.image_to_string(img, config='--psm 6'))


plt.show()
