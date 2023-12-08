import time
import os,shutil
'''
def get_pest_prediction(*args):
    return 'hs'
def get_recommended_pesticide(*args):
    return ["fdsafdhgoas"]
'''
from .Recommend import get_recommended_pesticide
from .DetectPest import get_pest_prediction
def get_pesticide(user):
    user.is_processing = True
    pests = []
    
    #for image in something
    for image_path in os.listdir(os.path.join("temp",user.token)):
        pests.append(get_pest_prediction(os.path.join(os.path.join("temp",user.token),image_path)))    
    
    pesticide_pests = {}
    for pest in pests:
        pesticide_pests[pest] = get_recommended_pesticide(pest,user.history_of_purchases)
    
    user.pesticide_pests = pesticide_pests
    user.is_processing = False