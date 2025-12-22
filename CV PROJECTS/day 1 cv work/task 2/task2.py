import cv2
import numpy as np
import matplotlib.pyplot as plt
gray=cv2.imread('grayscale.jpg')
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
mean=np.mean(gray)
median=np.median(gray)
std_dev=np.std(gray)
images=[gray,hist]
titles=['Grayscale Image','Histogram']
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,img,title in zip(axes,images,titles):
    if title=='Histogram':
        ax.plot(img)
        ax.set_xlim([0,256])
    else:
        ax.imshow(img,cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.text(0.5, -0.1, f'Mean: {mean:.2f}, Median: {median:.2f}, Std Dev: {std_dev:.2f}', fontsize=10)
plt.savefig('output.png')
plt.show()
