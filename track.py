import cv2
import numpy as np
import os

# อ่านภาพ
path = 'dataset/text binary'
for file in os.listdir(path):
    if file.endswith(('.jpg', '.jpeg', '.png', '.JPG', 'HEIC', '.PNG')):
        file_path = path+"/"+file
        print(file_path)
        image = cv2.imread(file_path)

        # แปลงเป็นภาพขาวดำ (Grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # ใช้ Adaptive Threshold เพื่อแยกตัวอักษร
        thresh = cv2.threshold(gray, 255, 255, 
                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 

        # ใช้ Morphological Operations เพื่อลดการเชื่อมต่อของตัวอักษร
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # thresh = cv2.bitwise_not(thresh)

        # kernel = np.ones((2, 2), np.uint8) 
        # thresh = cv2.erode(thresh, kernel, iterations=1) 

        # ค้นหา Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))

        # คัดกรองเฉพาะ contour ที่มีขนาดใหญ่พอและเล็กเกินไป
        MIN_AREA = 100  # พื้นที่ขั้นต่ำ
        MAX_AREA = 8000  # พื้นที่สูงสุด
        filtered_contours = [cnt for cnt in contours if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA]

        name = file_path.split("/")[2]
        count = 0
        # วนลูปตีกรอบแต่ละตัวอักษร
        for i, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            object_img = thresh[y:y + h, x:x + w]
            # cv2.imshow(f'Character {i+1}', object_img)
            contours1, _ = cv2.findContours(object_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        	# คำนวณพื้นที่ของแต่ละ Contour
            areas = [cv2.contourArea(cnt) for cnt in contours1]

        	# หาพื้นที่ที่ใหญ่ที่สุด
            max_area = max(areas)

        	# สร้างหน้ากาก (Mask) ใหม่เพื่อเก็บเฉพาะกลุ่มที่ใหญ่ที่สุด
            mask = np.zeros_like(object_img)

            for i, contour in enumerate(contours1):
        	    if cv2.contourArea(contour) == max_area:  # ตรวจสอบพื้นที่เท่ากับพื้นที่ใหญ่ที่สุด
        	        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # เติมสีขาวใน Mask

        	# นำ Mask มาซ้อนทับกับภาพต้นฉบับ
            result = cv2.bitwise_and(object_img, object_img, mask=mask)  

            cv2.imwrite("dataset/text/"+str(count)+name, result)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # ตีกรอบบนภาพต้นฉบับ
            print("save : dataset/text/"+str(count)+name)
            count += 1

# แสดงภาพผลลัพธ์ทั้งหมด
# cv2.imshow('Detected Characters', image)
# cv2.imshow('Detected morphed', thresh)
# cv2.imshow('Detected gray', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
