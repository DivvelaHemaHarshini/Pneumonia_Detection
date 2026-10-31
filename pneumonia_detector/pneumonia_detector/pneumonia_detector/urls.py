from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from classifier import views as classifier_views
from lungsound_classifier import views as lungsound_views # Ensure this is imported

urlpatterns = [
    path('admin/', admin.site.urls),
    
    path('', classifier_views.home_view, name='home'),
    
    # X-Ray App URLs
    path('xray/classify/', classifier_views.classify_image, name='classify_image'),
    path('xray/history/', classifier_views.prediction_history_view, name='prediction_history'),
    
    # X-Ray Report URL
    path(
        'report/<str:result>/<str:lang>/', 
        classifier_views.generate_report_view, 
        name='generate_report'
    ),
    
    # Lung Sound App URLs
    path('sound/classify/', lungsound_views.classify_sound_view, name='classify_sound'),
    path('sound/history/', lungsound_views.sound_history_view, name='sound_history'),
    
    # NEW Lung Sound Report URL
    path(
        'sound/report/<str:result>/<str:lang>/',
        lungsound_views.generate_sound_report_view,
        name='generate_sound_report'
    ),
    
    # Allauth URLs
    path('accounts/', include('allauth.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)