from django.apps import AppConfig
from rest_framework.response import Response
from django.conf import settings
import json
import datetime

ERROR_CODE = 399
SUCCESS_CODE = 200

class FaceRecognizeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'face_recognize'

####################################################################################################

class ErrorResponse(Response):
    def __init__(self, message):
        printt(message)
        Response.__init__(self,
            {'Error': message},
            status=ERROR_CODE, content_type="application/json")

####################################################################################################

class SuccessResponse(Response):
    def __init__(self, message):
        Response.__init__(self,
            {'Success': message},
            status=SUCCESS_CODE, content_type="application/json")

####################################################################################################

class JsonResponse(Response):
    def __init__(self, jsonString):
        # Nếu jsonString đã là dict, không cần gọi json.loads()
        if isinstance(jsonString, str):
            json_data = json.loads(jsonString)
        else:
            json_data = jsonString
        Response.__init__(self,
            json_data,
            status=SUCCESS_CODE, content_type="application/json")

####################################################################################################

class ObjResponse(Response):
    def __init__(self, jsonObj):
        Response.__init__(self,
            jsonObj,
            status=SUCCESS_CODE, content_type="application/json")

####################################################################################################

def utcnow():
    return datetime.datetime.utcnow()

####################################################################################################

def GetVNtime():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=7)

####################################################################################################

def RequireParamExist(request, param, paramName=None):
    value = GetParam(request, param)
    if not IsValid(value):
        if paramName is None:
            paramName = param
        raise Exception("Thiếu tham số " + paramName)
    return value

####################################################################################################

def IsParamExist(request, param):
    value = GetParam(request, param)
    return IsValid(value)

####################################################################################################

def RequireLevel(jwt, levels):
    if jwt["level"] not in levels:
        raise Exception("Bạn phải đăng nhập để có thể thao tác")
    return True

####################################################################################################

def GetParam(request, param, defaultValue=""):
    params = request.POST if len(request.POST) > 0 else request.data    
    return params.get(param, defaultValue)

####################################################################################################

def printt(msg):
    if settings.DEBUG:
        print(">>>>" + str(msg))

####################################################################################################

def IsValid(val):
    return val is not None and val != ""

####################################################################################################

def IsPk(_pk):
    return _pk and len(_pk) == 24 and " " not in _pk

####################################################################################################

def WriteLog(_activity, _exception):
    try:
        log = Log(
            activity=_activity,
            exception=_exception,
            timeCreate=utcnow()
        )
        log.save()
        return True    
    except Exception as e:
        return False
