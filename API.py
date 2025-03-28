from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
import webbrowser
import os

encode = {'_n':'-น',
          '_m':'-ม',
          '_o':'-อ',
          'K':'ก',
          'KH':'ฃ',
          'C':'ค',
          'NG':'ง',
          'J':'จ',
          'CH':'ฉ',
          'NN':'ณ',
          'D':'ด',
          'T':'ต',
          'N':'น',
          'B':'บ',
          'PA':'ป',
          'PH':'ผ',
          'F':'ฝ',
          'P':'พ',
          'M':'ม',
          'Y':'ย',
          'R':'ร',
          'L':'ล',
          'V':'ว',
          'S':'ส',
          'H':'ห',
          'HL':'หลฯ',
          'OY':'อยฯ',
          'A':'ะ',
          'Aa':'ั',
          'AAA':'า',
          'EI':'ิ',
          'EE':'ี',
          'EU':'ุ',
          'EA':'เ',
          'AI':'ใ',
          }

train_dir = "dataset\\model\\train\\"
val_dir = 'dataset\\model\\test\\'
target_img_shape = (70,70)
nclass = 35

train_datagen = ImageDataGenerator()
train_set = train_datagen.flow_from_directory(train_dir,target_size=target_img_shape,batch_size=32,class_mode = 'categorical')

val_datagen = ImageDataGenerator()
val_set = val_datagen.flow_from_directory(val_dir,shuffle=False,target_size=target_img_shape,batch_size=32,class_mode = 'categorical')

labels =(train_set.class_indices)
labels = dict((v,k) for k,v in labels.items())

model = load_model('Model/OCR-lanna.h5')
webbrowser.open('file://' + os.path.realpath('template/OCR-test.html'))
app = Flask(__name__, template_folder='template')
CORS(app)

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files.get('image')
    file_bytes = image_file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    new_height = 150 
    h, w = image.shape[:2]
    new_width = int((new_height / h) * w)

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    white_pixels = np.sum(gray == 255)
    black_pixels = np.sum(gray == 0)

    if white_pixels < black_pixels:
        print("binary")
        thresh = cv2.threshold(gray, 255, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
        kernel = np.ones((2, 2), np.uint8)
        dilated_image = cv2.dilate(thresh, kernel, iterations=2)

        thresh = cv2.erode(dilated_image, kernel, iterations=2)

    else :
        print("Not binary")
        thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1] 
        thresh = cv2.bitwise_not(thresh)


    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA = 40 
    MAX_AREA = 8000 
    filtered_contours = [cnt for cnt in contours if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA]

    count = 0
    obj_Y = []
    obj_b = []

    predicts = []

    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        object_img = thresh[y:y + h, x:x + w]
        
        if h < 10 or w < 20 : continue

        contours1, _ = cv2.findContours(object_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours1]
        max_area = max(areas)
        mask = np.zeros_like(object_img)

        for i, contour in enumerate(contours1):
            if cv2.contourArea(contour) == max_area: 
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        result = cv2.bitwise_and(object_img, object_img, mask=mask)      
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        result = cv2.resize(result,(70,70))
        img_array = img_to_array(result)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        predict = model.predict(img_array)
        pred_cls = labels[np.argmax(predict, axis=-1)[0]]
        print("prediction: ", encode[pred_cls])

        # ========================== UPDATE ==================================
        similaly = []
        path_check = os.path.join(train_dir,pred_cls)
        print(path_check)
        for file_image in os.listdir(path_check) :
            # print(file_image)
            path_image = os.path.join(path_check,file_image)
            template = cv2.imread(path_image)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template = cv2.resize(template,(70,70))
            results = cv2.matchTemplate(template, cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(results)
            # print(max_val)
            similaly.append(max_val)
        avg_similaly = max(similaly)
        # print(avg_similaly)
        # ====================================================================

        object_center = [x+(w//2), y+(h//2)]
        # cv2.imshow(f'Character {count}', result)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, pred_cls, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.circle(image, (object_center[0],object_center[1]), 2, (0,0,255), -1)
        obj_Y.append(y)
        obj_b.append(y+h)
        predicts.append([encode[pred_cls],object_center,x,x+w,avg_similaly])
        count += 1

    under_line = int(sum(obj_b)/len(obj_b))+6
    line = int(sum(obj_Y)/len(obj_Y))-9
    cv2.line(image, (0,line), (5000,line), (0, 0, 255), 1)
    cv2.line(image, (0,under_line), (5000,under_line), (0, 0, 255), 1)

    top_vowel = []
    under_vowel = []
    alphabet = []
    for i in range(len(predicts)):
        if predicts[i][1][1] < line :
            if predicts[i][0] in ['ค','ง','ั','ิ','ี','เ','ก','ม','ป','ย','พ','ว']:
                top_vowel.append(predicts[i])
            continue

        if predicts[i][1][1] > under_line :
            if predicts[i][0] in ['เ','า','ใ','ิ','ี','ะ','ร']:
                predicts[i][0] = 'ุ'
            under_vowel.append(predicts[i])
            continue
        
        else :
            alphabet.append(predicts[i])

    alphabet.sort(key = lambda x : x[1][0])
    top_vowel.sort(key = lambda x : x[1][0])
    under_vowel.sort(key = lambda x : x[1][0])
    sort_similaly = []
    alp = []
    text = ""
    # print(top_vowel)
    # print(alphabet)
    # print(under_vowel)

    for i in range(len(alphabet)):
        if alphabet[i][0] in ['ิ','ี'] :
            text += 'ค'
            alp.append('ค')
        else :
            text += alphabet[i][0]
            sort_similaly.append(alphabet[i])
            alp.append(alphabet[i][0])

        for j in range(len(top_vowel)):
            if alphabet[i][2] < top_vowel[j][2] < alphabet[i][3] or alphabet[i][2] < top_vowel[j][1][0] < alphabet[i][3] :
                if top_vowel[j][0] in ['ค','ง','เ','ก','ม','ป','ย','พ','ว'] :
                    text += 'ิ'
                    alp.append('ิ')
                else :
                    text += top_vowel[j][0]
                    alp.append(top_vowel[j][0])
                sort_similaly.append(top_vowel[j])
                top_vowel[j][0] = ""

        for j in range(len(under_vowel)):
            if alphabet[i][2] < under_vowel[j][2]+5 < alphabet[i][3] or alphabet[i][2] < under_vowel[j][1][0] < alphabet[i][3] :
                text += under_vowel[j][0]+"ฯ"
                alp.append(under_vowel[j][0]+"ฯ")
                sort_similaly.append(under_vowel[j])
                under_vowel[j][0] = ""

    if "-" in text:text = text.replace("-","")

    text_edit = ""
    for t in range(len(text)):
        text_edit += text[t]
        if t == 0 or t == len(text)-1: continue
        if text[t] in ['ั','ิ','ี','ุ']:
            if text[t+1] in ['ั','ิ','ี','ุ','เ','ใ','า','ะ']:
                text_edit += 'น'

        if text[t] in ['ั','า','ิ','ี','ุ']:
            if text[t-1] == ['ั']:
                text_edit += 'น'

    if "ฯ" in text_edit:text_edit = text_edit.replace("ฯ","")
    print(text)
    print(text_edit)
    similarities = [pred[4]*100 for pred in sort_similaly]
    print(alp)
    print(similarities)

    return jsonify({'text_LN': text, 'text': text_edit, 'similarities': similarities, 'alp': alp})

if __name__ == '__main__':
    app.run()
