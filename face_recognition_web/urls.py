from django.urls import path
from face_recognize import views
from face_recognize.views import home
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', home, name='index'),
    path('get_recognition_history/', views.get_recognition_history, name='get_recognition_history'),
    path('export-recognition-history/', views.export_recognition_history, name='export_recognition_history'),
    path('clear_recognition_history/', views.clear_recognition_history, name='clear_recognition_history'),
    path('recognize_video/', views.recognize_video, name='recognize_video'),
    path("stop-camera/", views.stop_camera, name="stop_camera"),
    path("api/save-new-face", views.save_new_face, name="save_new_face"),
    path('api/getNewID', views.GenerateID, name='getNewID')
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)