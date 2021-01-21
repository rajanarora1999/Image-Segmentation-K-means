#import reqd libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
#read the image using open cv
name_of_img=input("Enter the name of image(Case Sensitive)=")
name_of_img+=".jpg"
im=cv2.imread(name_of_img)
#open cv reads in BGR so convert to RGB
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.title("This is the original Image")
plt.imshow(im)
plt.axis('off')
plt.show()
#flatten the image
#we flatten all the pixels, now number of pixels =no of data points and no of features=3 for each data point i.e RGB
all_pixels=im.reshape((-1,3))
dominant_colors=int(input("Enter the number of Dominant Colors="))
km=KMeans(n_clusters=dominant_colors)
km.fit(all_pixels)
#this will give me the dominant  colors
colors=km.cluster_centers_
plt.figure(0,figsize=(100,100))
i=1
for each_color in colors:
    plt.subplot(1,dominant_colors,i)
    plt.axis('off')
    i+=1
    a=np.zeros((100,100,3),dtype='uint8')
    a[:,:,:]=each_color
    plt.imshow(a)
plt.title("The dominant Colors are :")
plt.show()
#now take an empty array of same size and assign colors to pixels on the basis of cluster it belongs to
new_img=np.zeros(all_pixels.shape,dtype='uint8')
for i in range(new_img.shape[0]):
    new_img[i]=colors[km.labels_[i]]
new_img=new_img.reshape(im.shape)
plt.imshow(new_img)
plt.axis('off')
plt.title("The segmented image is")
plt.show()
