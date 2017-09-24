import matplotlib.pyplot as plt
import math

import cv2
import os

image_folder = 'series'
video_name = 'video.mp4'
if not os.path.exists(video_name):
    os.makedirs(video_name)

#circle1 = plt.Circle((0, 0), 1, color='black',linewidth=1,fill=False)
fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot



#strangth one point SP;
SP_L=[];
for i in range (100):
    SP_L.append((i,i**2/100))

for i in range (100):
    SP_L.append((i,100-i**2/60)

index=0;   
for SP in SP_L:
    c = plt.Circle(SP, 1, color='red')
    ax.add_artist(c)
    #circle center point
    CP = (0,0)

    #DST between SP and CP
    def getDST():
        return math.sqrt((SP[0]-CP[0])**2 + (SP[1]-CP[1])**2) 

    for i in range(1,130):
        for j in range(-20,20):
            CP = (0,j)
            c = plt.Circle(CP, i+getDST()%1, color='black',linewidth=0.1,fill=False)
            ax.add_artist(c)
            
    ax.set_xlim((-100, 100))
    ax.set_ylim((-100, 100))
    fig.set_size_inches(9, 9)
    #plt.show()
    index=index+1;
    fig.savefig('series/' + str(index).zfill(5)+'.png')
    ax.clear();


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 24, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
