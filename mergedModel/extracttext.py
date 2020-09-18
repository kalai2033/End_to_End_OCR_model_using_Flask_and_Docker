import textboxgeneration as tb
import cv2
#import pandas as pd
import numpy as np
from itertools import compress
import networkx as nx
from PIL import Image
import argparse
import goal
#import time
#import os


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def common_data(list1, list2): 
    result = False
    count=0
    threshold = (0.75 * (min(len(list1),len(list2))))
    for x in list1: 
        for y in list2: 
    
            # if one common 
            if x == y: 
                count+=1
    
     
    if count > threshold:   
        result = True                
    
    return result

def set_image_dpi(i,im):
    #im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save('/home/bilal/Downloads/GenieFiles/DockerSetup/Models/myOCR/gray/'+str(i)+'.jpg', dpi=(300,300))
    
def clusterandsortboxes(poly):
    #df=pd.read_csv(path_filename,header=None, delimiter=',', names=['x1','y1','x2','y2','x3','y3','x4','y4'])
    boxes=sorted(poly, key=lambda r:r[1])
    area=[]
    center=[]
    for i,(x1,y1,x2,y2,x3,y3,x4,y4) in enumerate(boxes):
        minY = min([y1,y2,y3,y4])
        maxY = max([y1,y2,y3,y4])
        area.append([minY,maxY,(maxY-minY)])    
        startX = min([x1,x2,x3,x4])
        startY = min([y1,y2,y3,y4])
        endX = max([x1,x2,x3,x4])
        endY = max([y1,y2,y3,y4])
        height = endY - startY
        width = endX - startX    
        x=startX+(width/2)
        y=startY+(height/2)    
        center.append([x,y])
    group=[]
    for i in range(len(area)):  
        curr=np.arange(area[i][0], area[i][1])
        overlap=[]    
        for j,(minY,maxY,height) in enumerate(area):        
            comp=np.arange(minY, maxY)
            if common_data(curr, comp):
                if abs(center[i][1]-center[j][1]) < (min(area[i][2],area[j][2]))/2:
                    overlap.append(j)        
        group.append(overlap)

    for i,lis in enumerate(group):
        if i not in lis:
             lis.append(i)

    supersets = list(map(lambda a: list(filter(lambda x: len(a) < len(x) and set(a).issubset(x), group)), group))
    new_list = list(compress(group, list(map(lambda x: 0 if x else 1, supersets))))
    b_set = set(tuple(x) for x in new_list)
    b = [ list(x) for x in b_set ]
    b.sort(key = lambda x: new_list.index(x))
    G = nx.Graph()
    #Add nodes to Graph    
    G.add_nodes_from(sum(b, []))
    #Create edges from list of nodes
    q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in b]
    for i in q:
        #Add edges to Graph
        G.add_edges_from(i)
    #Find all connnected components in graph and list nodes for each component
    finallist=[list(i) for i in nx.connected_components(G)]        


    neewlist=[]
    for member in finallist:
        if len(member) < 2:
            neewlist.append(member)
            continue
        xval=[]
        for m in member:
            xval.append(boxes[m][0])
        com=(np.asarray(member)[np.argsort(xval)]).tolist()
        neewlist.append(com) 

    #flat_list = [item for sublist in neewlist for item in sublist]       
    #sortboxes=boxes[np.asarray(flat_list)]     
    
    return boxes, neewlist
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='test/', help='path to image file')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')

    args = parser.parse_args()
    
    
    
    poly=tb.loadmodel(args.cuda,"/home/bilal/Downloads/GenieFiles/DockerSetup/Models/myOCR/t5.jpg")
   
  
    
    orig=cv2.imread("/home/bilal/Downloads/GenieFiles/DockerSetup/Models/myOCR/t5.jpg")
    
    boxes,clusters=clusterandsortboxes(poly)
    

    for i,(x1,y1,x2,y2,x3,y3,x4,y4) in enumerate(boxes):
        startX = min([x1,x2,x3,x4])
        startY = min([y1,y2,y3,y4])
        endX = max([x1,x2,x3,x4])
        endY = max([y1,y2,y3,y4])
        height = endY - startY
        width = endX - startX
        padding_x = int(0.01 * width)
        padding_y = int(0.01 * height)    
        if padding_y < 2:
            padding_y=2    
        if padding_x<2:
            padding_x=2        
        if padding_x > startX :
            padding_x=startX    
        if padding_y > startY :
            padding_x=startY    
        #roi = orig[startY:endY, startX:endX]   
        roi = orig[startY-padding_y:endY+2*padding_y, startX-padding_x:endX+2*padding_x] 
        set_image_dpi(i,Image.fromarray(roi))    
        #cv2.imwrite('gray/'+str(i)+'.jpg',roi)
      
    
    model,demo_loader=goal.load_model()
    oplist=goal.extract_text(model,demo_loader)
    for line in clusters:
        textrowwise=[]
        for element in np.asarray(line):
            textrowwise.append(oplist[element])
        print(*textrowwise)
        
    
    
    
    
    
