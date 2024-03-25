import os
import cv2
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
import numpy as np

input_dir = '/home/johnadi/Desktop/computer-vision-tutorials-yt/QRreader+attendance-system/data/'

for j in sorted(os.listdir(input_dir)):
    img = cv2.imread(os.path.join(input_dir, j))
    qr_info =  decode(img)
    #print(qr_info)
    #print(type(qr_info))
    #print(j, len(qr_info))
    
    for qr in qr_info:
        
        data = qr.data
        rect = qr.rect
        polygon = qr.polygon
        
        print(data)
        print(rect)
        print(polygon)
        
        img = cv2.rectangle(img, (rect.left, rect.top), 
                            (rect.left + rect.width, rect.top + rect.height), 
                            (0,255,0), 5)
        
        img = cv2.polylines(img, [np.array(polygon)], True, (255,0,0), 5)
        
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()