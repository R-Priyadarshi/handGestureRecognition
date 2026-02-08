import os
import cv2

rootDirectory = 'Testing'
if not os.path.exists(rootDirectory):
    os.makedirs(rootDirectory)

noOfLettersToTrain = 26
noOfImagesPerLetter = 100

camera = cv2.VideoCapture(0)
camera.set(3,1280)
camera.set(4,640)
for label in range(noOfLettersToTrain):
    if not os.path.exists(os.path.join(rootDirectory, str(label))):
        os.makedirs(os.path.join(rootDirectory, str(label)))

    print(f"Collecting Images for {chr(label+65)}")
    while True:
        success, imgFromCam = camera.read()
        cv2.putText(imgFromCam, f'Press "Q" to collect images for {chr(label+65)} ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('GestureSphere - Ready to Capture',imgFromCam )
        cv2.moveWindow('GestureSphere - Ready to Capture',350,100)
        key=cv2.waitKey(25)
        if key == ord('q'):
            break
        if key == ord('w'):
            camera.release()
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyAllWindows()
    counter = 0
    while counter < noOfImagesPerLetter:
        success, imgFromCam = camera.read()
        cv2.imshow(f'GestureSphere - Capturing {chr(label+65)}', imgFromCam)
        cv2.moveWindow(f'GestureSphere - Capturing {chr(label+65)}',350,100)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(rootDirectory, str(label), f'{counter}.jpg'), imgFromCam)
        counter += 1
    cv2.destroyAllWindows()

camera.release()
cv2.destroyAllWindows()