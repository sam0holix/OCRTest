import cv2 as cv
import argparse
import pytesseract
import numpy as np
from Preprocess import platePreProcess

parser = argparse.ArgumentParser(description='plate detection EAST and Tesseract')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', required=True,
                    help='Path to a binary .pb file of model contains trained weights.')
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.')
parser.add_argument('--nms',type=float, default=0.4,
                    help='Non-maximum suppression threshold.')
parser.add_argument('--pad',type=float, default=0.06,
                    help='padding')
args = parser.parse_args()

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            if(score < scoreThresh):
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            cosA = np.cos(angle)
            sinA = np.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / np.pi))
            confidences.append(float(score))

    return [detections, confidences]

def recognize(rois):
    results = []
    conf = '-l eng --oem 1 --psm 7'
    for roi in rois:
        frame = platePreProcess(roi)
        gray = frame.gray()
        blur = frame.gaussianBlur((5,5))
        #_,thresh = frame.threshBinaryOTSU(blur)
        text = pytesseract.image_to_string(blur,config=conf)
        results.append(text)
        cv.imshow('roi',gray)
        cv.waitKey(0)
    text = "".join(results).strip()
    return text

def main():
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model
    padding = args.pad

    net = cv.dnn.readNet(model)

    kWinName = "Plate Detection"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outNames = []
    outNames.append("feature_fusion/Conv_7/Sigmoid")
    outNames.append("feature_fusion/concat_3")

    url = 'http://IP:PORT/video' #ip video feed possible
    cap = cv.VideoCapture(args.input if args.input else url)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        cv.imshow(kWinName,frame)
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        net.setInput(blob)
        outs = net.forward(outNames)
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)
        rois = []
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        for i in indices:
            vertices = cv.boxPoints(boxes[i[0]])
            
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            
            X1 = int(vertices[0][0])
            Y1 = int(vertices[1][1])
            X2 = int(vertices[3][0])
            Y2 = int(vertices[3][1])
            dX = int((X2 - X1) * padding)
            dY = int((Y2 - Y1) * padding)
            X1 = max(0,X1 - dX)
            Y1 = max(0, Y1 - dY)
            X2 = min(width_,X2+(dX*2))
            Y2 = min(height_,Y2+(dY*2))
            print(X1,Y1,X2,Y2)
            rois.append(frame[Y1:Y2,X1:X2])
    text = recognize(rois)
    print(text)
            
            
        
        

if __name__ == "__main__":
    main()
