import cv2
import numpy as np
image = cv2.imread('fire.jpg')
if image is None:
    print("Image not found!")
    exit()
image = cv2.resize(image, (640, 480))
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_fire = np.array([18, 150, 150])
upper_fire = np.array([35, 255, 255])
mask = cv2.inRange(hsv, lower_fire, upper_fire)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=4)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fire_found = False
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1500:
        fire_found = True
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "FIRE DETECTED", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
if fire_found:
    print("Fire detected in image!")
else:
    print("No fire detected.")
cv2.imshow("Fire Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()