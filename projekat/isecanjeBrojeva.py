import cv2
import numpy as np
from scipy import ndimage
from vector import distance, pnt2line
from matplotlib.pyplot import cm 
import itertools
import time
from skimage import exposure

cc = -1
def nextId():
    global cc
    cc += 1
    return cc

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal



def izdvojBrojeve(nazivVideo, x1z, y1z, x2z, y2z, x1p, y1p, x2p, y2p):

    kernel = np.ones((2,2),np.uint8)
    lower = np.array([230, 230, 230])
    upper = np.array([255, 255, 255])

    elements = []
    t =0
    counterZ = 0
    counterP = 0
    times = []
    passedDigitsZ = []
    passedDigitsP = []

    cap = cv2.VideoCapture(nazivVideo)

    while(cap.isOpened()):
    
        ret, img = cap.read()

        if ret:
            #l = np.array(lower, dtype = "uint8")
            #u = np.array(upper, dtype = "uint8")
            #mask = cv2.inRange(img, l, u)    
            #img0 = 1.0*mask

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img0 = cv2.threshold(gray, 167, 255, 0)

            img0 = cv2.dilate(img0,kernel) #cv2.erode(img0,kernel)
            img0 = cv2.dilate(img0,kernel)
            labeled, nr_objects = ndimage.label(img0)
            objects = ndimage.find_objects(labeled)
           
            for i in range(nr_objects):
                loc = objects[i]
                (xc,yc) = ((loc[1].stop + loc[1].start)/2,
                        (loc[0].stop + loc[0].start)/2)
                (dxc,dyc) = ((loc[1].stop - loc[1].start),
                        (loc[0].stop - loc[0].start))

                if(dxc>11 or dyc>11):
                    cv2.circle(img, (xc,yc), 16, (25, 25, 255), 1)
                    elem = {'center':(xc,yc), 'size':(dxc,dyc), 't':t}
                    # find in range
                    lst = inRange(20, elem, elements)
                    nn = len(lst)
                    if nn == 0:
                        elem['id'] = nextId()
                        elem['t'] = t
                        elem['passZ'] = False
                        elem['passP'] = False
                        elem['history'] = [{'center':(xc,yc), 'size':(dxc,dyc), 't':t}]
                        elem['future'] = [] 
                        #img0 = cv2.erode(img0, np.ones((2,2),np.uint8))
                        #img0 = cv2.erode(img0, np.ones((2,2),np.uint8))
                        isecena = img0[yc-dyc/2 : yc+dyc/2, xc-dxc/2 : xc+dxc/2]                        
                        resized = cv2.resize(isecena, (28, 28))
                        #resized = cv2.erode(resized, np.ones((3,3),np.uint8))
                        resized = cv2.erode(resized, np.ones((4,4),np.uint8))
                        elem['digitContour'] = resized
                        elements.append(elem)
                    elif nn == 1:
                        lst[0]['center'] = elem['center']
                        lst[0]['t'] = t
                        lst[0]['history'].append({'center':(xc,yc), 'size':(dxc,dyc), 't':t}) 
                        lst[0]['future'] = [] 
                    

            for el in elements:
                tt = t - el['t']
                if(tt<3):
                    linijaZ = [(x1z,y1z), (x2z, y2z)]
                    distZ, pntZ, rZ = pnt2line(el['center'], linijaZ[0], linijaZ[1])
                    cz = (25, 25, 255)
                    if rZ>0:
                        cv2.line(img, pntZ, el['center'], (0, 255, 25), 1)
                        if(distZ<9):
                            cz = (0, 255, 160)
                            if el['passZ'] == False:
                                el['passZ'] = True
                                counterZ += 1
                                passedDigitsZ.append(el['digitContour'])

                  
                    #cv2.circle(img, el['center'], 16, cz, 2)

                    id = el['id']
                    #cv2.putText(img, str(el['id']), 
                    #    (el['center'][0]+10, el['center'][1]+10), 
                    #    cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                    for hist in el['history']:
                        ttt = t-hist['t']
                    #    if(ttt<100):
                    #        cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                    for fu in el['future']:
                        ttt = fu[0]-t
                    #    if(ttt<100):
                    #        cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

                
                if(tt<3):
                    linijaP = [(x1p,y1p), (x2p, y2p)]
                    distP, pntP, rP = pnt2line(el['center'], linijaP[0], linijaP[1])
                    cp = (25, 25, 255)
                    if rP>0:
                        cv2.line(img, pntP, el['center'], (255, 0, 25), 1)
                        if(distP<9):
                            cp = (0, 255, 160)
                            if el['passP'] == False:
                                el['passP'] = True
                                counterP += 1
                                passedDigitsP.append(el['digitContour'])

                  
                    #cv2.circle(img, el['center'], 16, cp, 2)

                    id = el['id']
                    #cv2.putText(img, str(el['id']), 
                    #    (el['center'][0]+10, el['center'][1]+10), 
                    #    cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                    for hist in el['history']:
                        ttt = t-hist['t']
                    #    if(ttt<100):
                    #        cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                    for fu in el['future']:
                        ttt = fu[0]-t
                    #    if(ttt<100):
                    #        cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

            #elapsed_time = time.time() - start_time
            #times.append(elapsed_time*1000)
            #cv2.putText(img, 'Z: '+str(counterZ), (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(90,90,255),2)
            #cv2.putText(img, 'P: '+str(counterP), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(90,90,255),2)       

            #print nr_objects
            t += 1
            #if t%10==0:
            #    print t
                        

            #cv2.imshow('bla', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
             break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return passedDigitsZ, passedDigitsP



