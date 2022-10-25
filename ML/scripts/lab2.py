import scipy
import numpy
import matplotlib
import pandas
import sklearn

def countOnes(arr):
    count = 0;
    for x in arr:
        if x == 1:
            count+=1
    return count
# Large Purchase,Electronics,Card Type,Referred,Fraud
data=pandas.read_csv(".\..\data\\fraud_detection_data.txt", header=None);
# print(data[0].value_counts())
# print(data[1].value_counts())
# print(data[2].value_counts())
# print(data[3].value_counts())
#
# print(data[4].value_counts())

#print(data.groupby(level=0).groups())
# df.groupby('Team').filter(lambda x: len(x) >= 3)

#print(data[0].groupby(level=0, by=["0"]).count())
# print(len(data.groupby([0]).get_group(0))) # counts the number of class 0 in col 0
#
# print((data.groupby([0]).get_group(1)))
# print("# groupby col large purchase")
# print((data.groupby([1]).get_group(1)))
# print("# groupby col electronics")
# print((data.groupby([2]).get_group(1)))
# print("groupby col card type")
#print(data.groupby([3]).get_group(1))
#print((data.groupby([0]).get_group(1))[4]) #length is the number of 1's in col 0
class1attr0col4= data.groupby([0]).get_group(1)[4]
class1attr1col4=data.groupby([1]).get_group(1)[4]
class1attr2col4=data.groupby([2]).get_group(1)[4]
class1attr3col4=data.groupby([3]).get_group(1)[4]
onesInCol0 = len(class1attr0col4)
onesInCol1 = len(class1attr1col4)
onesInCol2 = len(class1attr2col4)
onesInCol3 = len(class1attr3col4)
#print(class1attr0col4)
onesGroupedByAttr0 = countOnes(class1attr0col4)
onesGroupedByAttr1 = countOnes(class1attr1col4)
onesGroupedByAttr2 = countOnes(class1attr2col4)
onesGroupedByAttr3 = countOnes(class1attr3col4)

print(onesGroupedByAttr0,onesGroupedByAttr1,onesGroupedByAttr2,onesGroupedByAttr3 )
oneRatioAttr0=float(onesGroupedByAttr0/onesInCol0)
oneRatioAttr1=float(onesGroupedByAttr1/onesInCol1)
oneRatioAttr2=float(onesGroupedByAttr2/onesInCol2)
oneRatioAttr3=float(onesGroupedByAttr3/onesInCol3)

print(oneRatioAttr0,oneRatioAttr1,oneRatioAttr2, oneRatioAttr3)
# print((data.groupby([3]).get_group(1))[4])
# print("groupby col Referred")


#spit the data on attr 0
attr0is0 = data.groupby([0]).get_group(0)
attr0is1 = data.groupby([0]).get_group(1)
#split on other attributes to find the best


attr0is0split1 = attr0is0.groupby([1]).get_group(0)[4]
attr0is0split2 = attr0is0.groupby([2]).get_group(1)[4]
attr0is0split3 = attr0is0.groupby([3]).get_group(1)[4]
print(countOnes(attr0is0split1), countOnes(attr0is0split2), countOnes(attr0is0split3))
print(type(attr0is0))
print(attr0is0split1)
attr0is1spli1  = attr0is1.groupby([1]).get_group(1)[4]
attr0is1spli2  =attr0is1.groupby([2]).get_group(1)[4]
attr0is1spli3  = attr0is1.groupby([3]).get_group(1)[4]

print(countOnes(attr0is1spli1), countOnes(attr0is1spli2), countOnes(attr0is1spli3))
