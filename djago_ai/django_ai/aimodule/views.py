from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
import os
# import matplotlib.pyplot as plt 
from PIL import Image , ImageDraw 
from sklearn.preprocessing import * 
import time 
import ast 
import os 
import numpy as np
from keras.models import load_model
import pandas as pd 
import matplotlib.pyplot as plt


kor_eng_dic_hard_coding = {"stairs":"계단","pig":"돼지","octopus":"문어","onion":"양파","pants":"바지","pencil":"연필","airplane":"비행기","blueberry":"블루베리","bush":"풀숲","cloud":"구름","canoe":"카누","clock":"시계","hat":"모자","hammer":"해머","firetruck":"소방차","tooth":"치아","mouth":"입","owl":"부엉이","piano":"피아노","saw":"톱","windmill":"풍차",
"strawberry":"딸기","pineapple":"파인애플","bear":"곰","butterfly":"나비","cow":"소","eraser":"지우개","angel":"천사","beard":"수염","book":"책","compass":"컴파스","cookie":"쿠키","crown":"왕관","headphones":"헤드폰","hand":"손","map":"지도","matches":"성냥","lipstick":"립스틱","paintbrush":"붓","picture frame":"액자","scissors":"가위","wheel":"바퀴",
"streetlight":"가로등","potato":"감자","apple":"사과","camera":"카메라","crab":"게","eyeglasses":"안경","ant":"개미","bed":"침대","boomerang":"부메랑","computer":"컴퓨터","couch":"소파","ear":"귀","helicopter":"헬리콥터","keyboard":"키보드","purse":"지갑","tornado":"토네이도","nail":"손톱","pillow":"베개","radio":"라디오","scorpion":"전갈","snowflake":"눈송이",
"tiger":"호랑이","rabbit":"토끼","backpack":"백팩","car":"자동차","crocodile":"악어","fish":"물고기","arm":"팔","harp":"하프","bracelet":"팔찌","castle":"성","diamond":"다이아몬드","elbow":"팔꿈치","foot":"발","knee":"무릎","sink":"싱크대","moon":"달","necklace":"목걸이","pizza":"피자","rain":"비","screwdriver":"드라이버","soccer ball":"축구공",
"toothbrush":"칫솔","raccoon":"너구리","banana":"바나나","carrot":"당근","cup":"컵","flower":"꽃","axe":"도끼","belt":"혁대","brain":"뇌","cell phone":"휴대폰","dishwasher":"식기세척기","envelope":"봉투","fork":"포크","ladder":"사다리","skull":"해골","mosquito":"모기","nose":"코","pond":"연못","rainbow":"무지개","wine glass":"와인잔","snowman":"눈사람",
"toothpaste":"치약","river":"강","bee":"벌","cat":"고양이","duck":"오리","frog":"개구리","bandage":"붕대","bread":"빵","cactus":"선인장","cello":"첼로","dog":"개","floor lamp":"램프","hot dog":"핫도그","hamburger":"햄버거","skyscraper":"빌딩","train":"기차","table":"테이블","parachute":"낙하산","remote control":"리모콘","sheep":"양","sock":"양말",
"traffic light":"신호등","shark":"상어","bicycle":"자전거","chair":"의자","elephant":"코끼리","giraffe":"기린","baseball bat":"야구방망이","broccoli":"브로콜리","cake":"케이크","church":"교회","dolphin":"돌고래","face":"얼굴","house":"집","leaf":"나뭇잎","fence":"울타리","tree":"나무","octagon":"팔각형","passport":"여권","paper clip":"클립","zebra":"얼룩말","spider":"거미",
"watermelon":"수박","shoe":"신발","hospital":"병원","kangaroo":"캥거루","horse":"말","grapes":"포도","bat":"박쥐","bucket":"양동이","calculator":"계산기","circle":"원","donut":"도넛","feather":"깃털","ice cream":"아이스크림","leg":"다리","finger":"손가락","triangle":"삼각형","sword":"칼","peanut":"땅콩","panda":"판다","star":"별","spoon":"숟가락",
"whale":"고래","snail":"달팽이","key":"열쇠","laptop":"노트북","lion":"사자","microphone":"마이크","bathtub":"욕조","bird":"새","calendar":"달력","candle":"양초","toe":"발가락","drill":"드릴","jacket":"재킷","light bulb":"전구","grass":"잔디","umbrella":"우산","t-shirt":"티셔츠","rhinoceros":"코뿔소","roller coaster":"롤러코스터","steak":"스테이크","sun":"태양",
"broom":"빗자루","snake":"뱀","monkey":"원숭이","mountain":"산","mushroom":"버섯","ocean":"바다","beach":"해변","bus":"버스","camel":"낙타","cannon":"대포","dragon":"용","crayon":"크레파스","jail":"감옥","lightning":"벼락","guitar":"기타","mouse":"쥐","oven":"오븐","penguin":"펭귄","sandwich":"샌드위치","submarine":"잠수함","swan":"백조"}


model = load_model('./model_dir/resnet_model.h5')

def make_img(img_arr) : 
    image = Image.new("P", (256,256), color=255) 
    image_draw = ImageDraw.Draw(image)
    for stroke in img_arr:
        for i in range(len(stroke[0])-1): 
            image_draw.line([stroke[0][i], 
            			stroke[1][i], 
                             	stroke[0][i+1], 
                             	stroke[1][i+1]], 
                             	fill=0, width=5)

    return image 


class_label =  ['cake', 'paintbrush', 'ear', 'sink', 'sock', 'nail', 'frog', 'paper clip', 'book', 'skull', 'candle', 'sun', 'hospital', 'river', 'pants', 'soccer ball', 'beach', 'headphones', 'church', 'brain', 'fish', 'piano', 'belt', 'moon', 'grass', 'toothbrush', 'hand', 'dragon', 'diamond', 'ant', 'foot', 'pond', 'computer', 'finger', 'tree', 'ladder', 'scissors', 'penguin', 'strawberry', 'cactus', 'helicopter', 'chair', 'rainbow', 'microphone', 'bear', 'bee', 'sheep', 'tiger', 'map', 'necklace', 'clock', 'cell phone', 'camera', 'pencil', 't-shirt', 'oven', 'bathtub', 'axe', 'horse', 'knee', 'harp', 'jail', 'baseball bat', 'floor lamp', 'whale', 'beard', 'apple', 'compass', 'traffic light', 'leg', 'submarine', 'mushroom', 'snowflake', 'flower', 'leaf', 'couch', 'fence', 'envelope', 'mountain', 'bandage', 'mouse', 'elephant', 'shark', 'pig', 'spider', 'lipstick', 'pizza', 'hammer', 'tooth', 'hot dog', 'bird', 'snowman', 'lion', 'dolphin', 'streetlight', 'scorpion', 'umbrella', 'stairs', 'cookie', 'wine glass', 'angel', 'canoe', 'wheel', 'hamburger', 'train', 'banana', 'sandwich', 'crab', 'calendar', 'lightning', 'crown', 'radio', 'dog', 'cow', 'kangaroo', 'toothpaste', 'mosquito', 'cloud', 'castle', 'bat', 'passport', 'guitar', 'carrot', 'feather', 'rabbit', 'bush', 'table', 'nose', 'star', 'cup', 'grapes', 'cannon', 'windmill', 'drill', 'calculator', 'rhinoceros', 'bread', 'fork', 'skyscraper', 'zebra', 'bus', 'boomerang', 'light bulb', 'broom', 'blueberry', 'remote control', 'circle', 'butterfly', 'bicycle', 'laptop', 'ice cream', 'shoe', 'key', 'firetruck', 'house', 'panda', 'pineapple', 'elbow', 'raccoon', 'octagon', 'crayon', 'cello', 'watermelon', 'camel', 'peanut', 'eyeglasses', 'owl', 'potato', 'picture frame', 'arm', 'backpack', 'pillow', 'swan', 'matches', 'giraffe', 'bed', 'ocean', 'car', 'keyboard', 'rain', 'bracelet', 'face', 'duck', 'toe', 'jacket', 'monkey', 'roller coaster', 'donut', 'sword', 'mouth', 'hat', 'crocodile', 'dishwasher', 'spoon', 'snail', 'purse', 'steak', 'airplane', 'eraser', 'cat', 'parachute', 'bucket', 'onion', 'snake', 'screwdriver', 'triangle', 'door', 'octopus', 'tornado', 'broccoli', 'saw']
# class_label = ['airplane', 'alarm clock', 'ambulance', 'angel', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack']


@api_view(['POST'])
def ai_post(request):
    draw = request.data['drawing']
    width = request.data['canvasWidth']
    height = request.data['canvasHeight']
    # for문 3번 쓰는거 numpy 활용해서 바꿀 생각 하기.
    for i in range(len(draw)):
        for j in range(len(draw[i])):
            for k in range(len(draw[i][j])):
                if j%2 == 0:
                    draw[i][j][k] = int(draw[i][j][k]*256//width)
                else:
                    draw[i][j][k] = int(draw[i][j][k]*256//height)
                    
    img = make_img(draw)

    # plot 을 위해
    # plt.imshow(img)
    # plt.show()

    img = np.array(img.resize((64,64))) 
    img = img.reshape(64,64,1)

    # 콘솔에 print 좀 해달라고 해서 

    # print_array = (model.predict(np.array([img]))*100)
    # print(class_label)
    # print_array =np.round(print_array, 1)[0]
    # print_array = list(map(str, print_array))
    # for i in range(len(print_array)):
    #     print_array[i] += '%'
    # print(print_array)
    # print(class_label[np.argmax(model.predict(np.array([img])))])
    # print('Response :' ,{'predictionType' : 0, 'firstPrediction' : prediction_first_label , 'secondPrediction' : ''} )
    # print('Response :' ,{'predictionType' : 0, 'firstPrediction' : prediction_first_label , 'secondPrediction' : prediction_second_label} )
    # print('Response :' ,{'predictionType' : 2, 'firstPrediction' : '' , 'secondPrediction' : ''} )


    prediction_arr = model.predict(np.array([img]))[0]
    prediction_first_index = np.argmax(prediction_arr)
    prediction_second_index = np.argsort(prediction_arr, axis=0 )[-2]
    prediction_first_label = class_label[prediction_first_index]
    prediction_second_label = class_label[prediction_second_index]
    predict_prob_first = prediction_arr[prediction_first_index] * 100

    if kor_eng_dic_hard_coding.get(prediction_first_label):
        prediction_first_label = kor_eng_dic_hard_coding.get(prediction_first_label)

    if kor_eng_dic_hard_coding.get(prediction_second_label):
        prediction_second_label = kor_eng_dic_hard_coding.get(prediction_second_label)

    if predict_prob_first >= 70 :
        if prediction_first_label == 'cooler':
            prediction_first_label = prediction_second_label
        return Response(data={'predictionType' : 0, 'firstPrediction' : prediction_first_label , 'secondPrediction' : ''}, status=status.HTTP_200_OK)
    elif 70 > predict_prob_first >= 45:
        if prediction_first_label == 'cooler':
            prediction_first_label = prediction_second_label
        elif prediction_second_label == 'cooler':
            return Response(data={'predictionType' : 0, 'firstPrediction' : prediction_first_label , 'secondPrediction' : ''}, status=status.HTTP_200_OK)
        return Response(data={'predictionType' : 1, 'firstPrediction' : prediction_first_label , 'secondPrediction' : prediction_second_label}, status=status.HTTP_200_OK)
    elif 45> predict_prob_first:
        return Response(data={'predictionType' : 2, 'firstPrediction' : '' , 'secondPrediction' : ''}, status=status.HTTP_200_OK)
    else:
        return Response(status=status.HTTP_400_BAD_REQUEST)

# Create your views here.
