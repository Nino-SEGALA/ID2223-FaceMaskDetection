import numpy as np
import cv2


def predict_mask(img):
    # Use the given image as input, which needs to be blob(s).
    maskCNN.setInput(cv2.dnn.blobFromImage(img, size=(224, 224), swapRB=True, crop=False))
    # Runs a forward pass to compute the net output
    return maskCNN.forward()


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

# Load a model imported from Tensorflow
maskCNN = cv2.dnn.readNetFromTensorflow('Model/my_model.pb') #  'exported_pbtxt/output.pbtxt'

# load the pretrained face prediction model
FaceCNN = cv2.dnn.readNetFromCaffe(
        "Face_detection/deploy.prototxt",
        "Face_detection/res10_300x300_ssd_iter_140000.caffemodel")

threshold = 0.5

while True:
    ret, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    FaceCNN.setInput(blob)
    d = FaceCNN.forward()

    (h, w) = img.shape[:2]
    faces = []

    for i in range(d.shape[2]):
        confidence = d[0,0,i,2]
        if confidence > threshold:

            box = d[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # make sure the box is in the image
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            width = endX-startX
            higth = endY-startY

            if width>10 and higth>10:
                faces.append((startX, startY, width, higth))


    if len(faces)>0:
        for (x,y,w,h) in faces:
            prediction = predict_mask(img[y:y+h,x:x+w])
            if prediction>0.5:
                title = "MASK "+ str(int((prediction[0][0])*10000)/100) + "%"
                color = (0,255,0)
            else:
                title = "NO MASK!!! " + str(int((1-prediction[0][0])*10000)/100) + "%"
                color = (0,0,255)
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img, title, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
