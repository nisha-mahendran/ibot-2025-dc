import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
image=cv2.imread('frontview.jpg')
orb = cv2.ORB_create(nfeatures=500) # Detect up to 500 keypoints
# Detect keypoints and compute descriptors
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
keypoints, descriptors = orb.detectAndCompute(gray, None)
# Draw keypoints
result = cv2.drawKeypoints(image, keypoints, None,
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Match keypoints between two images
image2 = cv2.imread('sideview.jpg')
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
# Create BFMatcher (Brute Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors, descriptors2)
# Sort matches by distance (lower is better)
matches = sorted(matches, key=lambda x: x.distance)
match_img = cv2.drawMatches(image, keypoints, image2, keypoints2,matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Show results
images=[result, match_img]
titles=['ORB Keypoints','Matches']
fig,axes=plt.subplots(1,2,figsize=(8,8))
# row,column,figsize
#axes[0].imshow(image1) #for a specific subplot
for ax,img,title in zip(axes,images,titles):
    ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig('feature mathching.jpg')
plt.show()