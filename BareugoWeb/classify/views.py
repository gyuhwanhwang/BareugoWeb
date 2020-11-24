from django.contrib.auth.models import User
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import json

# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from .tmp_classify import classifier
from .Prediction import AI_Classifier

def index(request):
    return HttpResponse('Hello')

@csrf_exempt
def get_harmful_index(request):
    # data = {}
    # if request.method == "POST":
    #     print('HERE!!!!!!!')
    #     data = dict(request.POST)
    #     print(data)
    #     return JsonResponse(data)
    # else:
    #     print('json-data to be sent: ', data)
    #     return JsonResponse(data)
    data = {}
    if request.method == "POST":
        print('HERE!!!!!!!')
        data = json.loads(request.body)
        print(data)
        data = classifier(data)
        # data = AI_Classifier(data)
        return JsonResponse(data)
    else:
        print('json-data to be sent: ', data)
        return JsonResponse(data)