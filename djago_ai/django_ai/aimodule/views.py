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


class_label = ['airplane', 'alarm_clock', 'ambulance', 'angel', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack']

@api_view(['POST'])
def ai_post(request):
    draw = request.data['drawing']
    # for문 3번 쓰는거 numpy 활용해서 바꿀 생각 하기.
    for i in range(len(draw)):
        for j in range(len(draw[i])):
            for k in range(len(draw[i][j])):
                if j%2 == 0:
                    draw[i][j][k] = int(draw[i][j][k]*256//974)
                else:
                    draw[i][j][k] = int(draw[i][j][k]*256//1546)
                    
    img = make_img(draw)
    # plt.imshow(img)
    # plt.show()
    img = np.array(img.resize((64,64))) 
    img = img.reshape(64,64,1)
    # print(model.predict(np.array([img])))
    # print(class_label[np.argmax(model.predict(np.array([img])))])
    prediction_arr = model.predict(np.array([img]))[0]
    prediction_first_index = np.argmax(prediction_arr)
    # print(prediction_arr)
    prediction_second_index = np.argsort(prediction_arr, axis=0 )[-2]
    prediction_first_label = class_label[prediction_first_index]
    predict_prob_first = prediction_arr[prediction_first_index] * 100
    if predict_prob_first >= 80 :
        return Response(data={'predictionType' : 0, 'firstPrediction' : prediction_first_label , 'secondPrediction' : ''}, status=status.HTTP_200_OK)
    elif 80 > predict_prob_first >= 65:
        prediction_second_label = class_label[prediction_second_index]
        return Response(data={'predictionType' : 1, 'firstPrediction' : prediction_first_label , 'secondPrediction' : prediction_second_label}, status=status.HTTP_200_OK)
    elif 65> predict_prob_first:
        return Response(data={'predictionType' : 2, 'firstPrediction' : '' , 'secondPrediction' : ''}, status=status.HTTP_200_OK)
    else:
        return Response(status=status.HTTP_400_BAD_REQUEST)

# Create your views here.
