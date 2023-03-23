from django.urls import path
from . import views

urlpatterns = [
    path('audiopredict/', views.audio_post),
]