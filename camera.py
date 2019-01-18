import cv2
cam = cv2.VideoCapture(0)
import time

cv2.namedWindow("test1")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        for i in range(0,500):
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            time.sleep(.1)
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray_image, (200, 100))
            #cv2.imwrite(img_name, frame)
            cv2.imwrite(img_name, small)
            print("{} written!".format(img_name))
            img_counter += 1

cam.release()

cv2.destroyAllWindows()
