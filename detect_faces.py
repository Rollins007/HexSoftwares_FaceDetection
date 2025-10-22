# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
print("[INFO] loading image...")
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw thin green bounding box
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # display confidence inside the top-left corner of the box
        text = "{:.2f}%".format(confidence * 100)
        font_scale = 0.5
        thickness = 1

        # calculate text size to keep it within box boundaries
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_offset_x = startX + 4
        text_offset_y = startY + text_h + 2

        # ensure text fits inside face box
        if text_offset_y + text_h < endY:
            cv2.putText(image, text, (text_offset_x, text_offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        else:
            # fallback if face box is too small
            cv2.putText(image, text, (startX, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

# enhance brightness/contrast slightly for visual clarity
image = cv2.convertScaleAbs(image, alpha=1.15, beta=15)

# show the output in a larger resizable window
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", 600, 400)
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
