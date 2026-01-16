import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
image=cv2.imread('image.png')
height, width = image.shape[:2]
flipped_horizontal = cv2.flip(image, 1) #1 for horizontal flip 

#rotation transform
centre=(width//2, height//2) 
angle=45    
scale=1.0 
rotation_matrix=cv2.getRotationMatrix2D(centre, angle, scale)
rotated_image=cv2.warpAffine(image, rotation_matrix, (width, height))
#2)colour augmentation
#brightness adjustment
brightness_factor=1.2 #increase brightness by 20%
brighter = np.clip(image+(brightness_factor-1)*255, 0, 255).astype(np.uint8)


contrast_factor=1.5 #increase contrast by 50%
img_float=image.astype(np.float32)
contrasted=np.clip(128 + contrast_factor * (img_float - 128), 0, 255).astype(np.uint8)
contrasted_1=np.clip(img_float * contrast_factor, 0, 255).astype(np.uint8)

#saturation adjustment
image_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
saturation_factor=1.5 #increase saturation by 50%
image_hsv[:,:,1]=np.clip(image_hsv[:,:,1]*saturation_factor, 0, 255)
saturated_image=cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

images=[image,flipped_horizontal,rotated_image,brighter,contrasted,saturated_image]
titles=['Original Image','Flipped Horizontal','Rotated Image','Brighter Image','Contrasted Image','Saturated Image']
fig,axes=plt.subplots(2,3,figsize=(8,8))
for ax,img,title in zip(axes.flat,images,titles):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig('augmented_images.png')
plt.show()