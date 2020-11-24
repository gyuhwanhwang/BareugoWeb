from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    url(r'get_harmful_index/$', views.get_harmful_index, name='get_harmful_index'),
]