#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STUDENT_ID: 100193308
Created on: Sun Nov  6 20:54:27 2022
Last update:  Nov  22 2022, get gini and ig work
Description: implments the cw

"""


import math
#data 
data = [
        [1,0,1,0,1,0,1,0,0,0,1,0,0,1], #headache
        [1,1,0,1,1,1,0,0,1,1,0,1,0,0], #sports
        [1,0,1,1,1,1,1,1,0,0,0,1,0,0], #stiffneck
        [1,1,1,1,1,1,1,0,0,0,0,0,0,0]  #diagnosis
        ]
data1 = [
        [0,0,1,1,0,0,0,0,1,1,1,1], #A
        [0,0,0,1,0,0,0,0,0,0,0,1], #B
        [1,1,1,0,0,0,0,0,0,0,0,1], #C
        [0,0,0,0,1,1,1,1,1,1,1,1]  #Target
        ]

# input contingency table of the form ct = [[TN,FN],[FP,TP]]
#         diag B  M
# (ATTR1 -)   TN  FN  [[TN,FN],
# (ATTR1 +)   FP  TP  [FP,TP]]
def get_infomation_gain(ct):
    TN = ct[0][0]
    FN = ct[0][1]
    FP = ct[1][0]
    TP = ct[1][1]

    total = TN + FN + FP + TP
    
    
    if(total == 0):
        print("no data")
        return 0
    #sum by 'diagnosis' (FN+TP)/total  (TN+FP)/total
    entropy_at_root = -( (TN+FP)/total * math.log2((TN+FP)/total) + (FN+TP)/total * math.log2((FN+TP)/total ) )
    
    weight1,weight2,term1,term2,term3,term4,term5,term6=0,0,0,0,0,0,0,0
    if( TN!=0 and FN==0):
        weight1=(TN+FN)/total
        term1= -1 *( TN/(TN+FN)*math.log2(TN/(TN+FN)) )
        #wighted_sum+=((TN+FN)/total * -(TN/(TN+FN))*math.log2(TN/(TN+FN)))
    if( FN!=0 and TN==0):
        weight1=(TN+FN)/total
        term2= -1 *( FN/(TN+FN)*math.log2(FN/(TN+FN)) )
        #wighted_sum+=((TN+FN)/total * -(FN/(TN+FN))*math.log2(FN/(TN+FN)))
    if( FN!=0 and  TN!=0):
        weight1=(TN+FN)/total
        term3= -1 *( TN/(TN+FN)*math.log2(TN/(TN+FN)) + FN/(FN+TN)*math.log2(FN/(TN+FN)) )
        #wighted_sum+=( (TN+FN)/total * -(TN/(TN+FN)*math.log2(TN/(TN+FN)) + (FN/(TN+TN))*math.log2(FN/(TN+FN))) )
        
        
    if( TP!=0 and FP==0):
        weight2=(FP+TP)/total 
        term4=-1 *( FP/(FP+TP)*math.log2(FP/(FP+TP)) )
        #wighted_sum+=(FP+TP)/total * -((FP/(FP+TP))*math.log2(FP/(FP+TP))) 
    if( FP!=0 and TP==0):
        weight2=(FP+TP)/total 
        term5=-1 *( TP/(FP+TP)*math.log2(TP/(FP+TP)) )
        #wighted_sum+=((TN+FN)/total * -(FN/(TN+FN))*math.log2(FN/(TN+FN)))
    if( FP!=0 and  TP!=0):
        weight2=(FP+TP)/total 
        term6= -1 *( TP/(FP+TP)*math.log2(TP/(FP+TP)) + (FP/(TP+FP))*math.log2(FP/(TP+FP)) )
        #wighted_sum+=( (TN+FN)/total * -(TN/(TN+FN)*math.log2(TN/(TN+FN)) + (FN/(TN+TN))*math.log2(FN/(TN+FN))) )
            
        
    return entropy_at_root -( weight1*(term1+term2+term3) + weight2*(term4 + term5 + term6))
    
def get_gini(ct):
    TN = ct[0][0]
    FN = ct[0][1]
    FP = ct[1][0]
    TP = ct[1][1]
    total = TN + FN + FP + TP
    
    TNplusFNis0=False
    FPplusTPis0=False
    if(TN+FN==0):
        TNplusFNis0 = True
    if(FP+TP==0):
        FPplusTPis0 = True
    if(total == 0):
        print("no data")
        return 0
    
    gini_at_root = 1 - (math.pow((TP+FN)/total, 2)  + math.pow((FP+TN)/total,2))
    
    weight1,weight2,term1,term2,term3,term4=0,0,0,0,0,0
    
    if(not TNplusFNis0):
        weight1=(TP+FP)/total
        term1=math.pow((TP/(TP+FP)),2)
        term2=math.pow((FP/(TP+FP)), 2)
        gini_1 = 1 - (term1 +term2)
        #a=((TN+FN)/total * (1 - (math.pow(FN/(TN+FN),2) + math.pow(TN/(FN+TN), 2)))) 
    if(not FPplusTPis0):
        #b=((FP+TP)/total * (1 - (math.pow(FP/(FP+TP),2) + math.pow(TP/(FP+TP), 2))))
        weight2=(TN+FN)/total
        term3=math.pow(TN/(TN+FN),2)
        term4=math.pow(FN/(TN+FN), 2)
        gini_2 = 1-(term3 + term4)       
    return gini_at_root - weight1*gini_1 - weight2*gini_2

         #diag 0   1
# (ATTR1 -)   TN  FN  [[TN,FN],
# (ATTR1 +)   FP  TP  [FP,TP]]
def get_chi_squared(ct):
    TN = ct[0][0]
    FN = ct[0][1]
    FP = ct[1][0]
    TP = ct[1][1]
    
    total = TN + FN + FP + TP
    if(total == 0):
        print("no data")
        return 0
    expct_0_0 = ((TN+FP)/total) * ((FN+TN)/total) * total
    expct_0_1 = ((TN+FN)/total) * ((FN+TP)/total) * total
    expct_1_0 = ((TN+FP)/total) * ((FP+TP)/total) * total
    expct_1_1 = ((FP+TP)/total) * ((FN+TP)/total) * total

    return (math.pow((TN - expct_0_0),2) + math.pow((FN - expct_0_1),2) + math.pow((FP- expct_1_0),2)+math.pow((TP- expct_1_1),2))

def ct_from_data(data):

    attribute=data[2]
    class_attr=data[len(data)-1]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(data[0])):
        if (attribute[i] == 1 and class_attr[i] == 1):
                  tp+=1
        elif (attribute[i] == 0 and class_attr[i] == 0):
                  tn+=1
        elif (attribute[i] == 0 and  class_attr[i] == 1):
                  fn+=1
        elif (attribute[i] == 1 and  class_attr[i] == 0):
                  fp+=1
    return [[tn, fn],[fp, tp]]
    

def Headache_from_diagdnosis(data):
    headache=data[0]
    diagnosis=data[len(data)-1]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(data[0])):
        if (diagnosis[i] == 1 and headache[i] == 1):
                  tp+=1
        elif (diagnosis[i] == 0 and headache[i] == 0):
                  tn+=1
        elif (diagnosis[i] == 0 and  headache[i] == 1):
                  fn+=1
        elif (diagnosis[i] == 1 and  headache[i] == 0):
              fp+=1
    return [[tn, fn],[fp, tp]]

def Sports_from_diagdnosis(data):
    sports=data[1]
    diagnosis=data[len(data)-1]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(data[0])):
        if (diagnosis[i] == 1 and sports[i] == 1):
                  tp+=1
        elif (diagnosis[i] == 0 and sports[i] == 0):
                  tn+=1
        elif (diagnosis[i] == 0 and  sports[i] == 1):
                  fn+=1
        elif (diagnosis[i] == 1 and  sports[i] == 0):
              fp+=1
    return [[tn, fn],[fp, tp]]

def Stiffneck_from_diagdnosis(data):
    stiffneck=data[2]
    diagnosis=data[len(data)-1]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(data[0])):
        if (diagnosis[i] == 1 and stiffneck[i] == 1):
                  tp+=1
        elif (diagnosis[i] == 0 and stiffneck[i] == 0):
                  tn+=1
        elif (diagnosis[i] == 0 and  stiffneck[i] == 1):
                  fn+=1
        elif (diagnosis[i] == 1 and  stiffneck[i] == 0):
              fp+=1
    return [[tn, fn],[fp, tp]]

'''
h_ct_1 = Headache_from_diagdnosis(data)
IG_headahce=get_infomation_gain(h_ct_1)
Gini_headahce=get_gini(h_ct_1)
chi_squared_headache=get_chi_squared(h_ct_1)
print(f'IG_headache = {IG_headahce}' )
print(f'Gini_headache = {Gini_headahce}' )
print(f'chi_squared_headache = {chi_squared_headache}' )
s_ct_1 = Sports_from_diagdnosis(data)
IG_sports=get_infomation_gain(s_ct_1)
Gini_sports=get_gini(s_ct_1)
chi_squared_sports=get_chi_squared(s_ct_1)
print(f'IG_sports = {IG_sports}' )
print(f'Gini_sports = {Gini_sports}' )
print(f'chi_squared_sports = {chi_squared_sports}' )
sn_ct_1 = Stiffneck_from_diagdnosis(data)
IG_stiffneck=get_infomation_gain(sn_ct_1)
Gini_stiffneck=get_gini(sn_ct_1)
chi_squared_stiffneck=get_chi_squared(sn_ct_1)
print(f'IG_stiffneck= {IG_stiffneck}' )
print(f'Gini_stiffneck = {Gini_stiffneck}' )
print(f'chi_squared_stiffneck = {chi_squared_stiffneck}' )


with open('./../output/headache_splitting_diagnosis.txt', 'w') as headache_file:
    headache_file.write(f'IG_headache = {IG_headahce}\n')
    headache_file.write(f'Gini_headache = {Gini_headahce}\n')
    headache_file.write(f'chi_squared_headache = {chi_squared_headache}\n')
    
with open('./../output/sports_splitting_diagnosis.txt', 'w') as sports_file:
    sports_file.write(f'IG_sports = {IG_sports}\n' )
    sports_file.write(f'Gini_sports = {Gini_sports}\n' )
    sports_file.write(f'chi_squared_sports = {chi_squared_sports}\n' )

with open('./../output/stiffneck_splitting_diagnosis.txt', 'w') as stiffneck_file:
    stiffneck_file.write(f'IG_stiffneck= {IG_stiffneck}\n' )
    stiffneck_file.write(f'Gini_stiffneck = {Gini_stiffneck}\n' )
    stiffneck_file.write(f'chi_squared_stiffneck = {chi_squared_stiffneck}\n' )

'''
    
testdata=ct_from_data(data1)
IG_A=get_infomation_gain(testdata)
print(testdata)
IG_A=get_gini(testdata)
chi_squared_A=get_chi_squared(testdata)
print()
print(f'IG_A= {IG_A}' )
print(f'IG_A = {IG_A}' )
print(f'chi_squared_A = {chi_squared_A}' )
'''
testdata=ct_from_data(data1)
chi_squared_A=get_chi_squared(testdata)
print(f'chi_squared_A = {chi_squared_A}')
'''