from django.urls import path
from . import views

urlpatterns = [
    # The home page for image classification
    path('', views.classify_image, name='classify_image'),
    
    # The new page to view user-specific prediction history
    path('history/', views.prediction_history_view, name='prediction_history'),
]