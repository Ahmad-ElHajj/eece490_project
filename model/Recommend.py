import pandas as pd
import numpy as np
from pprint import pprint
import json
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
from math import log2

def extract_relevant_pesticides(pests):
    # Extracting relavent pests:
    df_pests = pd.read_csv("model/pests_to_pesticides.csv")
    df_pests = df_pests.fillna(0)

    # Contains the pesticide names
    headers= list(df_pests.columns)

    dataset_pests = df_pests.iloc[:,:1].values

    RESHAPE_pests= lambda x : list(x)[0]
    dataset_pests = list(map(RESHAPE_pests,dataset_pests))


    # Reshape list
    pests_indices=dict()

    for i in range(len(dataset_pests)):
        pests_indices[dataset_pests[i]]=i

    def relevant_pesticides(pests:list[str],df_pests)-> list:
        for pest in pests:
            assert pest in pests_indices, "[ Server ] : pest {} not found in database".format(pest)
        results={}
        for pest in pests:
            pest_index=pests_indices[pest]
            results[pest]=df_pests.columns[df_pests.iloc[pest_index] == 1].tolist()
        return results
    return relevant_pesticides(pests,df_pests)


def Recommend_Top_K_Pesticides(pesticides:list[str],user_history:dict,k:int):
    df_pesticides_main = pd.read_csv("model/pesticides.csv")

    # dropping columns
    columns_to_drop=["number_targeted_pests","price_oz","perc_act","oz_5acres","price_5acres"]
    df_pesticides = df_pesticides_main.drop(columns=columns_to_drop)

    headers= list(df_pesticides.columns)

    dataset_vectors    = df_pesticides.iloc[:,1:].values # here we are removing the name of pest

    dataset_pesticides = df_pesticides.iloc[:,:1].values
    # scale the vectors
    RESHAPE_pests= lambda x : list(x)[0]
    dataset_pesticides = list(map(RESHAPE_pests,dataset_pesticides))

    pesticides_indices=dict()

    for i in range(len(dataset_pesticides)):
        pesticides_indices[dataset_pesticides[i]]=i
    # print(df_pesticides)
    cosine_similarities = cosine_similarity(dataset_vectors, dataset_vectors)
    user_history= {dataset_pesticides[0]:1,dataset_pesticides[2]:23,dataset_pesticides[3]:1}
    user_pesticide_history=dict()
    def similarity_function(cos_sim: float,occurences:int):
        """ The function for similarity"""

        return cos_sim * log2(occurences)

    def combine(LIST):
        """ returns average of list"""
        return sum(LIST)/len(LIST)
    def get_sim_to_user_history(pesticide:str,user_history:dict):
        """ returns the similarity between pesticide and user_history. Values between -1 and 1 """
        assert pesticide in dataset_pesticides, "[ SERVER ] : Pesticide {} is not valid".format(pesticide)
        pesticide_index=pesticides_indices[pesticide]
        SIMILARITIES=[]
        for choice in user_history:
            choice_index=pesticides_indices[choice]
            sim=similarity_function(cosine_similarities[choice_index][pesticide_index],user_history[choice])
            SIMILARITIES.append(sim)
        return combine(SIMILARITIES)

    def recommend_top_k_pesticides(pesticides:list[str],user_history:dict,k:int):
        # assert k<= len(pesticides) , "[SERVER] : k = {} is greater than size of pesticides = {}".format(k,len(pesticides))
        SIMILARITIES=[(i,get_sim_to_user_history(pesticides[i],user_history)) for i in range(len(pesticides))]
        SIMILARITIES.sort(key=lambda x: x[1],reverse=True)
        return [pesticides[entry[0]] for entry in SIMILARITIES[0:k] ]

    return recommend_top_k_pesticides(pesticides,user_history,k)

def display_pesticide_information(df,pesticide:str,headers=[] ):
    """ returns information about pesticide in dictionary format. dataframe of the  """
    if len(headers)!= len(df.columns):
        headers=list(df.columns)
    row=list(df[df['Pesticide'] == pesticide].values[0])
    return dict (  [ (headers[i],row[i],) for i in range(len(row))] )



def get_recommended_pesticide(pest,user_history):
    pesticides = extract_relevant_pesticides([pest])
    return Recommend_Top_K_Pesticides(pesticides[pest],user_history,len(pesticides[pest]))

    # def load_datasets(name):
    #     df_pests = pd.read_csv(name)
    #     df_pests = df_pests.fillna(0)
    #     return df_pests
    

    # df_pests=load_datasets("/model/pests_to_pesticides.csv")
    # headers= list(df_pests.columns)
    # dataset_pests = df_pests.iloc[:,:1].values
    # RESHAPE_pests= lambda x : list(x)[0]
    # dataset_pests = list(map(RESHAPE_pests,dataset_pests))
    # pests_indices=dict()
    


    # for i in range(len(dataset_pests)):
    #     pests_indices[dataset_pests[i]]=i
    
    # def relevant_pesticides(pests:list[str],df_pests)-> list:
    #     for pest in pests:
    #         assert pest in pests_indices, "[ Server ] : pest {} not found in database".format(pest)
    #     results={}
    #     for pest in pests:
    #         pest_index=pests_indices[pest]
    #         results[pest]=df_pests.columns[df_pests.iloc[pest_index] == 1].tolist()
    #     return results

    # # get relevant pests
    # # get cosine similarities
    # # 
    # pass