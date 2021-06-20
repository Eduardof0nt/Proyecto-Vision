import cv2
import numpy as np
import yaml

# Read image 
img = cv2.imread('Img/test9.png',cv2.IMREAD_COLOR)

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lowR = (0,100,100)
highR = (10,255,255)
maskR = cv2.inRange(imgHSV, lowR, highR)

imgC = np.zeros(img.shape)
contours, hierarchy = cv2.findContours(maskR.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
valid_cntrs = []
for cntr in contours:
    peri = cv2.arcLength(cntr, True)
    approx = cv2.approxPolyDP(cntr, 0.04 * peri, True)
    if len(approx) == 4:
        valid_cntrs.append((cntr, approx))

valid_template_objects = []
for (c,approx0) in valid_cntrs:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = imgHSV[np.floor(y*0.95).astype('int'):np.ceil((y + h)*1.05).astype('int'), np.floor(x*0.95).astype('int'):np.ceil((x + w)*1.05).astype('int')]
    
    A = w*h
    
    lowG = (30,100,100)
    highG = (80,255,255)
    maskG = cv2.inRange(roi, lowG, highG)

    contoursG, hierarchy = cv2.findContours(maskG.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for cntr in contoursG:
        peri = cv2.arcLength(cntr, True)
        approx = cv2.approxPolyDP(cntr, 0.04 * peri, True)
        if cv2.countNonZero(maskG)/A > 0.2 and len(approx) > 5:
            valid_template_objects.append((x,y,w,h,approx0))

template_object = []
for i in valid_template_objects[0][4]:
    center = tuple(i[0])
    template_object.append(center)

template_object.sort(key=(lambda x: x[0] + x[1]))
if template_object[1][0] > template_object[2][0]:
    tmp = template_object[2]
    template_object[2] = template_object[1]
    template_object[1] = tmp

for i, center in enumerate(template_object):
    color = (255,0,0)
    if i == 3:
        color = (0,255,0)
    img = cv2.circle(img,center,10,color,thickness=-1)

s2 = img.shape*5
pts1 = np.float32(template_object)
pts2 = np.float32([[s2[0]-100,s2[1]-100],[s2[0]-100,s2[1]],[s2[0],s2[1]-100],[s2[0],s2[1]]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

result0 = cv2.warpPerspective(img, matrix, (int(s2[0]*2),int(s2[1]*2)),borderValue=(255,255,255))

gResult = cv2.cvtColor(result0,cv2.COLOR_BGR2GRAY)

_,bResult = cv2.threshold(gResult,250,255,cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(bResult.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

result = []
for cntr in contours:
    (x, y, w, h) = cv2.boundingRect(cntr)
    result = result0[np.floor(y).astype('int'):np.ceil((y + h)).astype('int'), np.floor(x).astype('int'):np.ceil((x + w)).astype('int')]

# Camera angles
cameraMatrix = []
distCoeff = []
with open("calibration_matrix.yaml", 'r') as stream:
    cameraCalibration = yaml.load(stream)
    cameraMatrix = np.array(cameraCalibration['camera_matrix'])
    distCoeff = np.array(cameraCalibration['dist_coeff'])
pts2Alt = np.zeros((4,3))
pts2Alt[:,0:2] = pts2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret,rvecs, tvecs = cv2.solvePnP(pts2Alt, pts1, cameraMatrix, distCoeff)

print(np.rad2deg(rvecs))
    
  

# def semiBin(x):
#     if x < 10:
#         return 0
#     else:
#         return x

# vectFunction = np.vectorize(semiBin)

# result2 = vectFunction(result)

# threshImg = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# contoursF, _ = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# for cntr in contoursF:
#     (x, y, w, h) = cv2.boundingRect(cntr)
#     result = cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)
#     cv2.putText(result, str('X'), (x + w//4, y + h//2),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    
# cv2.drawContours(image=result, contours=contoursF, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

# for v_o in valid_template_objects:
#     (x,y,w,h) = v_o
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# cv2.drawContours(image=imgC, contours=valid_cntrs, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


cv2.imshow("Image", img)
cv2.imshow("Resultado", result)
# cv2.imshow("Image", cv2.cvtColor(roi, cv2.COLOR_HSV2BGR))
# cv2.imshow("Mask Red", maskR)
cv2.waitKey(0)