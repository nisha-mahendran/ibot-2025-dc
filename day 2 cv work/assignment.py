import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
image=cv2.imread('coins.jpg')
gray_circles = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_circles, (15, 15), 0)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=400, param1=100, param2=80, minRadius=150, maxRadius=650) #adjust parameters accordingly
result_circles = image.copy()
sum=0
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(result_circles, (i[0], i[1]), i[2], (0, 255, 0), 20)
        # draw the center of the circle
        cv2.circle(result_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.putText(result_circles, f'R={i[2]}', (i[0]+10, i[1]), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 10)
        sum+=i[2]
    print(f"Detected {len(circles[0])} circles.")
    print(f"average radius {sum/len(circles[0])}")
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(result_circles, cv2.COLOR_BGR2RGB))
plt.savefig('result_circles.png')
plt.show()
plt.axis('off')  # Hide axes for better visualization

