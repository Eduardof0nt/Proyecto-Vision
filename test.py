# Python code for Multiple Color Detection


import numpy as np
import cv2


# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while(1):
    
	# Reading the video from the
	# webcam in image frames
    _, img = webcam.read()
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowR = (0,100,100)
    highR = (15,255,255)
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
        
        lowG = (42,20,20)
        highG = (100,255,255)
        maskG = cv2.inRange(roi, lowG, highG)
        cv2.imshow("ROI", maskG)
        contoursG, hierarchy = cv2.findContours(maskG.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        for cntr in contoursG:
            peri = cv2.arcLength(cntr, True)
            approx = cv2.approxPolyDP(cntr, 0.04 * peri, True)
            if cv2.countNonZero(maskG)/A > 0.2 and len(approx) > 5:
                valid_template_objects.append((x,y,w,h,approx0))

    template_object = []
    if len(valid_template_objects) > 0:
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

        pts1 = np.float32(template_object)
        pts2 = np.float32([[template_object[3][0]-100,template_object[3][1]-100],[template_object[3][0]-100,template_object[3][1]],[template_object[3][0],template_object[3][1]-100],list(template_object[3])])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        result = cv2.warpPerspective(img, matrix, (img.shape[0]*2,img.shape[1]*2))
                
        # Program Termination
        cv2.imshow("Resultado", result)
    cv2.imshow("Original", maskR)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break