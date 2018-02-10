import cv2
import numpy as np
import cPickle, gzip
import math
from isecanjeBrojeva import izdvojBrojeve
from sklearn.datasets import fetch_mldata
from knn import izmeniMnist, getKNN


def pronadjiZelenuLiniju(nazivVideo):

  cap = cv2.VideoCapture(nazivVideo)  
  ret, frame = cap.read()
  #cv2.imshow('Frame', frame)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  #kernel = np.ones((3, 3))
  opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

  zelena = opening.copy()
  zelena[:, :, 0] = 0
  zelena[:, :, 2] = 0

  grayZelena = cv2.cvtColor(zelena, cv2.COLOR_BGR2GRAY)
  #cv2.imshow('GrayZelena', grayZelena)

  edgesZelena = cv2.Canny(grayZelena , 50, 150, apertureSize = 3)
  #cv2.imshow('EdgesZelena', edgesZelena)

  linesZelena = cv2.HoughLinesP(edgesZelena, rho = 1, theta = np.pi / 180, threshold = 100, minLineLength = 100, maxLineGap = 10)
    
  #print 'Zelena: '
  #print linesZelena

  for linijaZ in linesZelena:
    for x1,y1,x2,y2 in linijaZ:
      cv2.line(zelena,(x1,y1),(x2,y2),(0,0,255),2)
      cv2.imwrite('houghlinesZ.jpg',zelena)

  #cv2.imshow('HoughZ', zelena)
  #cv2.waitKey(0)

  return linesZelena

def pronadjiPlavuLiniju(nazivVideo):
  
  cap = cv2.VideoCapture(nazivVideo)
  ret, frame = cap.read()
  #cv2.imshow('Frame', frame)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  #kernel = np.ones((3, 3))
  opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

  plava = opening.copy()

  plava[:, :, 1] = 0  
  plava[:, :, 2] = 0

  grayPlava = cv2.cvtColor(plava, cv2.COLOR_BGR2GRAY)
  #cv2.imshow('GrayPlava', grayPlava)

  edgesPlava = cv2.Canny(grayPlava , 50, 150, apertureSize = 3)
  #cv2.imshow('EdgesPlava', edgesPlava)
  
  linesPlava = cv2.HoughLinesP(edgesPlava, rho = 1, theta = np.pi / 180, threshold = 100, minLineLength = 50, maxLineGap = 10)

  #print 'Plava: '
  #print linesPlava

  for linijaP in linesPlava:
    for x1,y1,x2,y2 in linijaP:
      cv2.line(plava,(x1,y1),(x2,y2),(0,0,255),2)
      cv2.imwrite('houghlinesP.jpg',plava)

  #cv2.imshow('HoughP', plava)
  #cv2.waitKey(0)

  return linesPlava

def nadjiMinTacku(linije):

  listX = []
  listY = []

  for linija in linije:
    for x1, y1, x2, y2 in linija:
      listX.append(x1)
      listY.append(y1)

  #print min(listX)
  #print listY[listX.index(min(listX))]

  return min(listX), listY[listX.index(min(listX))]

def nadjiMaxTacku(linije):

  listX = []
  listY = []

  for linija in linije:
    for x1, y1, x2, y2 in linija:
      listX.append(x2)
      listY.append(y2)

  #print max(listX)
  #print listY[listX.index(max(listX))]

  return max(listX), listY[listX.index(max(listX))]


def nadjiJednacinuPrave(x1, y1, x2, y2):

  x = [x1, x2]
  y = [y1, y2]

  coeffs = np.polyfit(x, y, 1)
  
  k = coeffs[0]
  n = coeffs[1]
  #print k, n
  return  k, n

def iscrtajGraniceLinija(nazivVideo, x1z, y1z, x2z, y2z, x1p, y1p, x2p, y2p):

  cap = cv2.VideoCapture(nazivVideo)  
  ret, frame = cap.read()

  cv2.circle(frame, (x1z, y1z), 10, (0,255,255), -1)
  cv2.circle(frame, (x2z, y2z), 10, (0,255,255), -1)
  cv2.circle(frame, (x1p, y1p), 10, (0,255,255), -1)
  cv2.circle(frame, (x2p, y2p), 10, (0,255,255), -1)

  cv2.imshow('Granice', frame)
  cv2.waitKey(0)

def generisiZaglavljeIzlaza():

  f= open("out.txt","w+")
  f.write("RA 59/2014 Arsenije Degenek\r")
  f.write("file	sum\r")
  f.close() 

def generisiTeloIzlaza(zbir, brojac):
  f= open("out.txt","a+")
  f.write('video-' + str(brojac) + '.avi ' + str(zbir) + '\r')
  f.close()


def iseciBrojeve(nazivVideo):

  cap = cv2.VideoCapture(nazivVideo)
  frmcnt = 0
  while(cap.isOpened()):

    ret, frame = cap.read()
    digits = []
    if ret:

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(gray, 160, 255, 0)
      im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      if frmcnt % 1200 == 0 or frmcnt % 1990 == 0:
        for contour in contours:
          cords = cv2.boundingRect(contour)
          cv2.rectangle(frame, (cords[0], cords[1]) ,(cords[0] + cords[2], cords[1] + cords[3]), (0,0,255), 1)

          isecen = thresh[cords[1] : cords[1] + cords[3], cords[0] : cords[0] + cords[2]]
          cv2.imshow('Digit', isecen)
          cv2.waitKey(0)
        cv2.imshow('test', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
      frmcnt += 1
    else:
      break


def nadjiKonture(nazivVideo, listaTacaka):

  cap = cv2.VideoCapture(nazivVideo )
 
  prosleZelenu = []
  proslePlavu = []
  sliciceZelene = []
  slicicePlave = []
  frame_num = 0
  while(cap.isOpened()):

    ret, frame = cap.read()
    
    
    if ret:
      
      #if frame_num % 9 is not 0:
      #  frame_num += 1
      #  continue

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(gray, 160, 255, 0)
      im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      for contour in contours:
        cords = cv2.boundingRect(contour)
        
        xCentar = cords[0] + cords[2] / 2
        yCentar = cords[1] + cords[3] / 2
   
        distZ = linePointDistance(xCentar, yCentar, listaTacaka[0], listaTacaka[1], listaTacaka[2], listaTacaka[3])
        #print 'Udaljenost konture od zelene prave:' 
        #print distZ
        
        zelenaPrava = yCentar - (listaTacaka[6] * xCentar) - listaTacaka[7]
        plavaPrava = yCentar - (listaTacaka[8] * xCentar) - listaTacaka[9]

        #cv2.circle(frame,(listaTacaka[0], listaTacaka[1]), 10, (0,255,255), -1)
        #cv2.circle(frame,(listaTacaka[2], listaTacaka[3]), 10, (255,255,255), -1)
        
        if distZ <= 2 and distZ >= 0 and (xCentar >= listaTacaka[0]) and (xCentar <= listaTacaka[2]):  
          #print 'Zelena prava: '
          #print zelenaPrava
          ignorisem = False
          for pz in prosleZelenu:
            if (xCentar >= pz[0]) and (xCentar <= pz[0] + pz[2]) and (yCentar >= pz[1]) and (yCentar <= pz[1] + pz[3]):
              ignorisem = True
              break
          if not ignorisem:
            prosleZelenu.append(cords)
            slik = thresh[cords[1]:cords[1] + cords[3], cords[0]: cords[0] + cords[2]]
            sliciceZelene.append(slik)
            #cv2.circle(frame,(xCentar, yCentar), 10, (0, 255, 0), -1) 
          #print 'dira zelenu'
       
        if plavaPrava <= 4 and plavaPrava >= -4 and xCentar >= listaTacaka[4] and xCentar <= listaTacaka[5]:  
          #print 'Plava prava: '
          #print plavaPrava 
          proslePlavu.append(cords)
          cv2.circle(frame,(xCentar, yCentar), 10, (255, 0 ,0), -1) 
          #print 'dira plavu'

        #cv2.rectangle(frame, (cords[0], cords[1]) ,(cords[0] + cords[2], cords[1] + cords[3]), (0,0,255), 1)

      cv2.imshow('Konture', frame)
      frame_num += 1
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      break

  print 'Prosle zelenu: (%d)' % (len(prosleZelenu))
  print 'Prosle plavu: (%d)' % (len(proslePlavu))
  print len(sliciceZelene)
  cap.release()
  cv2.destroyAllWindows()

  return sliciceZelene
 



knn = getKNN()
#generisiZaglavljeIzlaza()
for i in range(0, 10):
  video = 'videos\\video-' + str(i) + '.avi'
#video = 'videos\\video-3.avi'
  linijeZ = pronadjiZelenuLiniju(video)
  linijeP = pronadjiPlavuLiniju(video)

  zelenaX1, zelenaY1 = nadjiMinTacku(linijeZ)
  zelenaX2, zelenaY2 = nadjiMaxTacku(linijeZ)

  plavaX1, plavaY1 = nadjiMinTacku(linijeP)
  plavaX2, plavaY2 = nadjiMaxTacku(linijeP)


  #iscrtajGraniceLinija(video, zelenaX1, zelenaY1, zelenaX2, zelenaY2, plavaX1, plavaY1, plavaX2, plavaY2)

  #kZelena, nZelena = nadjiJednacinuPrave(zelenaX1, zelenaY1, zelenaX2, zelenaY2)
  #kPlava, nPlava = nadjiJednacinuPrave(plavaX1, plavaY1, plavaX2, plavaY2)
  
  brojeviZ, brojeviP = izdvojBrojeve(video, zelenaX1, zelenaY1, zelenaX2, zelenaY2, plavaX1, plavaY1, plavaX2, plavaY2)


  zeleniZbir = 0
  plaviZbir = 0

  #print 'Zeleni: '


  for brojZ in brojeviZ:
    #cv2.imshow('digitZ', brojZ)
    #cv2.waitKey(0)
    num = int(knn.predict(brojZ.reshape(1, 784)))
    #print num
    zeleniZbir += num
    

  #print 'Plavi: '
  for brojP in brojeviP:
    #cv2.imshow('digitP', brojP)
    #cv2.waitKey(0)
    num = int(knn.predict(brojP.reshape(1, 784)))
    #print num
    plaviZbir += num
    

  #print 'Konacno: '
  #print zeleniZbir
  #print plaviZbir
  print 'Zbir za snimak %d :' % i
  print plaviZbir - zeleniZbir
  #generisiTeloIzlaza(konacan_zbir, i)


 

