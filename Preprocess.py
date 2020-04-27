import cv2 as cv 

class platePreProcess:


    def __init__(self, source):
        self.img = source

    def gray(self):
        self.gray = cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        return self.gray

    def gaussianBlur(self,value):
        self.blur = cv.GaussianBlur(self.gray,value,0)
        return self.blur
    
    def laplacianDerivative(self):
        self.laplacian = cv.Laplacian(self.blur,cv.CV_8U)
        return self.laplacian

    def sobelX(self):
        self.sobelX = cv.Sobel(self.blur,cv.CV_8U,1,0,ksize=3)
        return self.sobelX

    def sobelY(self):
        self.sobelY = cv.Sobel(self.blur,cv.CV_8U,0,1,ksize=3)
        return self.sobelY

    def threshBinaryOTSU(self,frame):
        self.frame = frame
        self.thresh = cv.threshold(self.frame,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return self.thresh

    def adaptiveThreshGaussian(self, frame):
        self.frame = frame
        self.thresh = cv.adaptiveThreshold(self.frame,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        return self.thresh

    def dilation(self,frame,kernel):
        self.frame = frame
        self.dilated = cv.dilate(self.frame,kernel,iterations=2)
        return self.dilated
    def morphOpen(self,frame,kernel):
        self.frame = frame
        self.morphOpen = cv.morphologyEx(self.frame,cv.MORPH_OPEN,kernel)
        return self.frame

    def morphClose(self,frame,kernel):
        self.frame = frame
        self.morphClose = cv.morphologyEx(self.frame,cv.MORPH_CLOSE,kernel)
        return self.frame

    def findContours(self,frame):
        self.frame = frame
        self.contours,self.hierarchy = cv.findContours(self.frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        return self.contours

