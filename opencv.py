import numpy as np
import cv2
import cv2.aruco as aruco
import math


# Function for finding ids and corners of aruco
def findaruco(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key =getattr(aruco,f'DICT_5X5_250')
    arucodict = aruco.Dictionary_get(key)
    arucoparam = aruco.DetectorParameters_create()
    (corners,ids,rejected) = cv2.aruco.detectMarkers(img,arucodict,parameters = arucoparam)
    aruco.drawDetectedMarkers(img,corners)
    return corners,ids

#reading arucomarkers 
img3 = cv2.imread("C://Users//piyus//Downloads//XD.jpg")
img4 = cv2.imread("C://Users//piyus//Downloads//LMAO.jpg")
img1 = cv2.imread("C://Users//piyus//Downloads//Ha.jpg")
img2 = cv2.imread("C://Users//piyus//Downloads//HaHa.jpg")

#straightening aruco markers
lis =[img1,img2,img3,img4]
roi = [0,0,0,0]
for i in range(4):
    c,d = findaruco(lis[i])
    print(c)
    tplft = (int(c[0][0][0][0]),int(c[0][0][0][1]))
    tprght = (int(c[0][0][1][0]),int(c[0][0][1][1]))
    btmlft = (int(c[0][0][3][0]),int(c[0][0][3][1]))
    btmrght = (int(c[0][0][2][0]),int(c[0][0][2][1]))
    cx = int((tplft[0]+btmrght[0])/2)
    cy = int((tplft[1]+btmrght[1])/2)
    tantheta = (btmrght[1]-btmlft[1])/(btmrght[0]-btmlft[0])
    angle = math.degrees(np.arctan(tantheta))
    print(angle)
    M =cv2.getRotationMatrix2D((cx,cy),angle,1.0)
    rotated = cv2.warpAffine(lis[i], M, (800,800))
    c,d = findaruco(rotated)
    print(d)
    tplft = (int(c[0][0][0][0]),int(c[0][0][0][1]))
    tprght = (int(c[0][0][1][0]),int(c[0][0][1][1]))
    btmlft = (int(c[0][0][3][0]),int(c[0][0][3][1]))
    btmrght = (int(c[0][0][2][0]),int(c[0][0][2][1]))
    
   # cropping arucomarkers 
    roi[i] = rotated[tplft[1]:btmlft[1],tplft[0]:tprght[0]]
   
    cv2.imshow("img",roi[i])
    cv2.imshow("img_1",rotated)    
    key = cv2.waitKey(0)  
    cv2.destroyAllWindows()



#resizing arucomarkers according to square size and putting them into list
lisst =[0,0,0,0]
lisst[0] = cv2.resize(roi[0],(178,178))
lisst[1] = cv2.resize(roi[1],(316,316))
lisst[2] = cv2.resize(roi[2],(210,210))
lisst[3] = cv2.resize(roi[3],(246,246))

for i in lisst: 
    cv2.imshow("img_4",i)       
    key = cv2.waitKey(0)  
    cv2.destroyAllWindows()


#reading the images with shapes and resizing it
img5 = cv2.imread("C://Users//piyus//Downloads//CVtask.jpg")
print(img5.shape)
img5 =cv2.resize(img5,(0,0),fx = 0.6,fy = 0.6)
cv2.imshow("shapes",img5)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

#coverting image to gray
gray = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY)

#finding canny image
canny = cv2.Canny(gray,30,150)

#finding contours
cont, hier = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

flag = 0
img_5 = img5.copy()
i = 0
print(img_5.shape)
for cnt in cont:
    if(flag%2 == 0):
        flag += 1
        continue
    flag += 1
    #detecting squares
    peri = cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,peri*0.02,True)
    if len(approx) == 4 :
        x,y,w,h = cv2.boundingRect(cnt)
        rat = w/float(h)
        if  rat >= 0.95 and rat <= 1.05:
            approx = np.reshape(approx,(4,2))
            lfttop = (int(approx[0,1]),int(approx[0,0]))
            rghttop = (int(approx[1,1]),int(approx[1,0]))
            rghtbtm = (int(approx[2,1]),int(approx[2,0]))
            lftbtm = (int(approx[3,1]),int(approx[3,0]))
            cx1 = int((lfttop[0]+rghtbtm[0])/2)
            cy1 = int((lfttop[1]+rghtbtm[1])/2)
            length=int( math.sqrt((rghttop[1]-lfttop[1])**2+(rghttop[0]-lfttop[0])**2))
            half =int(length/2)
            tantheta2 = ((rghtbtm[0]-lftbtm[0])/(rghtbtm[1]-lftbtm[1]))
            angle2 = math.degrees(np.arctan(tantheta2)) 
            print(angle2)
            pixel = img_5[cx1,cy1]

            #printing ids according to the colour of shapes
            # print(pixel)
            if np.all(pixel==[0,0,0]):
                print("id =3")
            elif np.all(pixel==[210,222,228]):
                print("id = 4")
            elif np.all(pixel==[9,127,240]):
                print("id = 2")
            else:
                print("id = 1")



            #making the squares black
            cv2.drawContours(img_5,[approx],-1,(0,0,0),-1)
            blank =np.zeros([w,h,3],dtype=np.uint8)


            #placing arucomarkers with their center aligned to black images of size of bounding rectangle
            blank[int((w/2)-half):int((w/2)+half),int((h/2)-half):int((h/2)+half)] = lisst[i]
            i+=1
            
            #rotating blank image with aruco to desired angle
            M =cv2.getRotationMatrix2D((int(w/2),int(h/2)),-angle2,1.0)
            rotated = cv2.warpAffine(blank, M, (w,h))
            # print(rotated.shape)
            # print(x,y,w,h)

            #adding the rotated arucomarkers into squares with bitwise or operation 
            img_5[y:y+h,x:x+w]=cv2.bitwise_or(rotated,img_5[y:y+h,x:x+w])
cv2.imshow("image",img_5)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

#saving the final image
cv2.imwrite("final.jpg",img_5)
