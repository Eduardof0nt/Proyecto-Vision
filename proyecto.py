import cv2
import numpy as np
import yaml

# Read image 
img = cv2.imread('Img/test12.png',cv2.IMREAD_COLOR)

cv2.imshow("Imagen Original", img)
h = np.histogram(img, bins=256, range=(0,255))[0]                                 # Cálculo de histograma

r_min = 0                                                               #Se calculan el minimo y máximo del histograma
r_max = 0
min_counter = 0
max_counter = 0
r = np.ceil(img.shape[0]*img.shape[1]*0.01) #1% de los píxeles más brillantes u obscuros
for i in range(0, h.shape[0]):
    if min_counter <= r or max_counter <= r:
        if h[i] != 0 and min_counter <= r:
            r_min = i
            min_counter += h[i]
        if h[h.shape[0]-i-1] != 0 and max_counter <= r:
            r_max = h.shape[0]-i-1
            max_counter += h[h.shape[0]-i-1]
    else:
        break

if r_max - r_min < 250:
    print('Se cambió el contraste de la imagen') #Alertar usuario
    print("Histograma de la imagen original desde {} hasta {}".format(r_min, r_max))
    m = 255.0/(r_max-r_min)
    img = (img*m)-r_min;
    img = np.clip(img, 0,255).astype(np.uint8)
    cv2.imshow("Imagen Ecualizada", img)

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
    
    lowG = (30,80,80)
    highG = (80,255,255)
    maskG = cv2.inRange(roi, lowG, highG)

    contoursG, hierarchy = cv2.findContours(maskG.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for cntr in contoursG:
        peri = cv2.arcLength(cntr, True)
        approx = cv2.approxPolyDP(cntr, 0.04 * peri, True)
        print(len(approx))
        if cv2.countNonZero(maskG)/A > 0.15 and len(approx) > 5:
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

angles = np.abs(np.rad2deg(rvecs[:,0]))
print('La posición de la cámara (ángulos) es la siguiente:')
print('Rotación en el eje "x": {0}°'.format(angles[0]))
print('Rotación en el eje "y": {0}°'.format(angles[1]))
print('Rotación en el eje "z": {0}°'.format(angles[2]))


# cv2.imshow("Image", img)
cv2.imshow("Resultado", result)
# cv2.imshow("Image", cv2.cvtColor(roi, cv2.COLOR_HSV2BGR))
# cv2.imshow("Mask Red", maskR)
cv2.waitKey(0)