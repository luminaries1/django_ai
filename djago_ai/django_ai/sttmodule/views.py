from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import AudioSerializer
from django.core.files.storage import FileSystemStorage
import random
from time import sleep


import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


# parser = argparse.ArgumentParser(description='KoSpeech')
# parser.add_argument('--model_path', type=str, required=False, default='./model/model.pt')
# parser.add_argument('--audio_path', type=str, required=False, default='../media/record.wav')
# parser.add_argument('--device', type=str, required=False, default='cpu')
# opt = parser.parse_args()

# opt = {''}



def handle_uploaded_file(f):
    # Save file to local directory
    filename = f.name
    fs = FileSystemStorage()
    fs.save(filename, f)
    # Return the file path
    return fs.path(filename)




# Create your views here.
@api_view(['POST'])
def audio_post(request):
    serializer = AudioSerializer(data=request.data)
    if serializer.is_valid():
        # handle the uploaded file
        audio_file = serializer.validated_data['audio_file']

        path = handle_uploaded_file(audio_file)
        

        # 예측 파트
        feature = parse_audio('home/ubuntu/aimodule/djago_ai/django_ai/media/record.wav', del_silence=True)
        input_length = torch.LongTensor([len(feature)])
        vocab = KsponSpeechVocabulary('home/ubuntu/aimodule/djago_ai/django_ai/sttmodule/label/ksponspeech_character_vocabs.csv')

        model = torch.load('home/ubuntu/aimodule/djago_ai/django_ai/sttmodule/model/model.pt', map_location=lambda storage, loc: storage).to('cpu')
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.eval()

        if isinstance(model, ListenAttendSpell):
            model.encoder.device = 'cpu'
            model.decoder.device = 'cpu'

            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, DeepSpeech2):
            model.device = 'cpu'
            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
            y_hats = model.recognize(feature.unsqueeze(0), input_length)

        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
        print(sentence)


        # ...
        dummy_word_lst = ['사과', '너구리', '수박', '쥐', '별', '번개', '달팽이' ]
        some_rand_int = random.randint(0,6)
        result_word = dummy_word_lst[some_rand_int]

        fs = FileSystemStorage()
        fs.delete(path)
        
        return Response(data = sentence)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)