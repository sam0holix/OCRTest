import cv2 as cv 
import numpy as np
from Preprocess import platePreProcess
import pytesseract
conf = '-l eng --oem 1 --psm 7'


def main():
    area = 10000
    source =  "IMG_20170825_154637.jpg"
    img = cv.imread(source)
    frame = platePreProcess(img)
    gray = frame.gray()
    blur = frame.gaussianBlur((5,5))
    sobelX = frame.sobelX()
    laplacian = frame.laplacianDerivative()
    _,thresh = frame.threshBinaryOTSU(sobelX)
    _,threshLaplace = frame.threshBinaryOTSU(laplacian)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(20,3))
    dilate = frame.dilation(thresh,kernel)
    contours = frame.findContours(dilate)
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        if ( w>h and cv.contourArea(contour) > area):
            im = gray[y:y+h,x:x+w]
            cv.imshow('contours',im)
            text = pytesseract.image_to_string(im,config=conf)
            print(text)
            cv.waitKey(0)
        
if __name__ == "__main__":
    main()