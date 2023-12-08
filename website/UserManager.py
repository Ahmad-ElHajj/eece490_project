class User:
    def __init__(self,token):
        self.token = token
        self.is_processing = False
        self.history_of_purchases = {}
        self.pesticide_pests = {}
    def __repr__(self):
        return str(self.token)
    
class Users:
    def __init__(self):
        self.users = {}
        try:
            with open(SAVEDUSERS,'rb') as svd:
                loaded = pickle.load(svd) 
                self.users = loaded.users
            print("User File was found!")
        except Exception:
            print("No User File was found.")
            
    def getUser(self,token):
        if token not in self.users:
            self.users[token] = User(token)
        return self.users[token]

    def display(self):
        return str(self.users)
    
    
    def save(self):
        with open(SAVEDUSERS,'wb') as svd:
            pickle.dump(self,svd)
    

import pickle
import sys
SAVEDUSERS = "SavedUsers.pkl"

users = Users()