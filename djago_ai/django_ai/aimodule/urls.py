from django.urls import path
from . import views

urlpatterns = [
    path('prediction/', views.ai_post),
]