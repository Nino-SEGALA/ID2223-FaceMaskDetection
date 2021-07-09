import cv2

def predict_mask(img):
    # Use the given image as input
    maskCNN.setInput(cv2.dnn.blobFromImage(img, size=(224, 224), swapRB=True, crop=False))
    # Runs a forward pass to compute the net output
    return maskCNN.forward()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

print("KOMMER HIT")

# load the mask prediction model
maskCNN = cv2.dnn.readNetFromTensorflow('Model/my_model.pb')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x,y,w,h) in faces:
        prediction = predict_mask(img[x:x+w,y:y+h])
        if prediction<0.5:
            title = "MASK"
            color = (0,255,0)
        else:
            title = "No Mask"
            color = (255,0,0)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img, title, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
