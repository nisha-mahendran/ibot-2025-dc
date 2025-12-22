import cv2
import matplotlib.pyplot as plt
image=cv2.imread('Lenna.png')
gaussian_blur=cv2.GaussianBlur(image,(7,7),0)
canny_edges=cv2.Canny(gaussian_blur,50,150)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
_,mask = cv2.threshold(gray,100,255,cv2.THRESH_BINARY) 
images=[image,gaussian_blur,canny_edges,mask]
titles=['Original Image','Gaussian Blur','Canny Edges','Binary Threshold']
fig,axes=plt.subplots(2,2,figsize=(8,8))
for ax,img,title in zip(axes.flat,images,titles):
    ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if len(img.shape)==3 else img,cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig('output.png')
plt.show()