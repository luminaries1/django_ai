from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import AudioSerializer
from django.core.files.storage import FileSystemStorage
import random
from time import sleep

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
        print(path)
        sleep(1)

        # ...
        dummy_word_lst = ['사과', '너구리', '수박', '쥐', '별', '번개', '달팽이' ]
        some_rand_int = random.randint(0,6)
        result_word = dummy_word_lst[some_rand_int]

        fs = FileSystemStorage()
        fs.delete(path)
        
        return Response(data = {result_word})
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)