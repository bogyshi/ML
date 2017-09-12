import sys
import csv
import pandas as pd
import math

class dTree(object):
    def __init__(self,data=None,a=True,c=0):
        self.left = None
        self.right = None
        self.data = data
        self.label = None
        self.isAt = a
        self.count = c

def bestAttribute(attributes,data):
	return 0

def id3(data,idclass,attributes):
	return 0

def parseData(filename):
	attributes=[]
	data=[]
	f = open(filename,'r')
	attributes=f.readline().split(",")
	attributes[-1]=attributes[-1][:-1] #removing newline character
	#print(attributes)
	#for l in f.readlines():
	#	print l
	df = pd.read_csv(filename)
	with open(filename,'rb') as csvFile:
		reader = csv.reader(csvFile,delimiter=',')
		for x in reader:
			data.append(x)
	return attributes,df

def infoGain(target,data,identifier):
        #calc entire set enropy
	net1 = data.where(data[identifier]==1)[identifier].count()
	net0 = data.where(data[identifier]==0)[identifier].count() 
        totnet = net1+net0
        pnet1=float(net1)/totnet
        pnet0=float(net0)/totnet
        resultnet = -1*pnet1*math.log(pnet1,2) + -1*pnet0*math.log(pnet0,2)
        #now etnropy of each variable in attribute TARGET
	first = data.where(data[target]==1) # only where attribute is 1
	second = data.where(data[target]==0) # only where attribute is 0
	pos1 = first.where(data[identifier]==1)[identifier].count() #where attribute is 1 and class is 1
	pos0 = first.where(data[identifier]==0)[identifier].count() # attribute 1 and class is 0
	nul1 = second.where(data[identifier]==1)[identifier].count() # attribute 0 and class 1
	nul0 = second.where(data[identifier]==0)[identifier].count() #attribute 0 and class 0
        
	tot1 = pos1+pos0
	tot2 = nul1+nul0

	p1=(float(pos1)/tot1)
	p2=(float(pos0)/tot1)
	n1=(float(nul1)/tot2)
	n2=(float(nul0)/tot2)
	
	resultp = -1*p1*math.log(p1,2) + -1*p2*math.log(p2,2)
	resultn = -1*n1*math.log(n1,2) + -1*n2*math.log(n2,2)
        c1=float(tot1)/totnet
        c0=float(tot2)/totnet

	return resultnet - (c1)*resultp - (c0)*resultn #info gain

def VI(target,data,identifier):
         #calc entire set vi
	net1 = data.where(data[identifier]==1)[identifier].count()
	net0 = data.where(data[identifier]==0)[identifier].count() 
        totnet = net1+net0
        pnet1=float(net1)/totnet
        pnet0=float(net0)/totnet
        vinet = net1*net0/(float(totnet)*totnet)

        first = data.where(data[target]==1) # only where attribute is 1
	second = data.where(data[target]==0) # only where attribute is 0
	pos1 = first.where(data[identifier]==1)[identifier].count() #where attribute is 1 and class is 1
	pos0 = first.where(data[identifier]==0)[identifier].count() # attribute 1 and class is 0
	nul1 = second.where(data[identifier]==1)[identifier].count() # attribute 0 and class 1
	nul0 = second.where(data[identifier]==0)[identifier].count() #attribute 0 and class 0
        
	tot1 = pos1+pos0
	tot2 = nul1+nul0

	p1=(float(tot1)/totnet)
	n1=(float(tot2)/totnet)
        
        resultp = pos1*pos0/(float(tot1)*tot1)
        resultn = nul1*nul0/(float(tot2)*tot2)

        return vinet - p1*resultp - n1*resultn


def id3Algo(df,identifier,attributes):
        root = dTree()
        c1 = df.where(df[identifier]==1)[identifier].count()
        c0 = df.where(df[identifier]==0)[identifier].count()
        if c1>c0:
            root.data=1
            root.count=c1
        else:
            root.data=0
            root.count=c0
        #check to see if there are any attributes left to split on, return leaf node if nothing left after determing majority leader
        if(len(attributes)==0):
            root.isAt=False
            return root
        mx=0.0
        mxVar = attributes[0]
        #determine variable of greatest info gain
        for x in attributes[:]:
	    ent = infoGain(x,data.filter(items=[x,identifier]),identifier)
            if(ent>mx):
                mx = ent
                mxVar = x

        # if info gain is 1, we have a pure subset, return leaf node
        if(mx==1):
            root.isAt=False
            return root

        attributes.remove(mxVar) # this is an inplace operation

        #case where there is no example in training set when current split takes 0 branch
        if(df.where(df[mxVar]==0)[identifier].count()==0):
            root.data=3
            
        #case where there is no example when we split with 1 branch
        if(df.where(df[mxVar]==1)[identifier].count()==0):
            if(root.data==3):
                root.data=5
            else:
                root.data=4

        if(root.data ==1 or root.data ==0):
            root.data=2
            
        #case when both splits result in no examples     
        if(root.data==5):
            root.isAt=False
            root.left=None
            root.right=None
            if c1>c0:
                root.data=1
                root.count=c1
            else:
                root.data=0
                root.count=c0
            return root
        else:
            root.label=mxVar
        #we either need to build the tree in both directions, or just to the right or left, or just return the leaf node.
        if(root.data==2):
            root.left=id3Algo(df.where(df[mxVar]==0),identifier,attributes[:])
            root.right=id3Algo(df.where(df[mxVar]==1),identifier,attributes[:])
            return root
        elif(root.data==3):
            if c1>c0:
                root.left = dTree(data=1,a=False,c=c1)
            else:
                root.left = dTree(data=0,a=False,c=c0)
            root.right=id3Algo(df.where(df[mxVar]==1),identifier,attributes[:])
            return root
        elif(root.data==4):
            if c1>c0:
                root.right= dTree(data=1,a=False,c=c1)
            else:
                root.right = dTree(data=0,a=False,c=c0)
            root.left=id3Algo(df.where(df[mxVar]==0),identifier,attributes[:])
            return root
        else:
            return root

def viAlgo(df,identifier,attributes):
        root = dTree()
        c1 = df.where(df[identifier]==1)[identifier].count()
        c0 = df.where(df[identifier]==0)[identifier].count()
        if c1>c0:
            root.data=1
            root.count=c1
        else:
            root.data=0
            root.count=c0
        #check to see if there are any attributes left to split on, return leaf node if nothing left after determing majority leader
        if(len(attributes)==0):
            root.isAt=False
            return root
        mx=0.0
        mxVar = attributes[0]
        #determine variable of greatest info gain
        for x in attributes[:]:
	    ent = VI(x,data.filter(items=[x,identifier]),identifier)
            if(ent>mx):
                mx = ent
                mxVar = x

        # if info gain is 1, we have a pure subset, return leaf node
        if(mx==1):
            root.isAt=False
            return root

        attributes.remove(mxVar) # this is an inplace operation

        #case where there is no example in training set when current split takes 0 branch
        if(df.where(df[mxVar]==0)[identifier].count()==0):
            root.data=3
            
        #case where there is no example when we split with 1 branch
        if(df.where(df[mxVar]==1)[identifier].count()==0):
            if(root.data==3):
                root.data=5
            else:
                root.data=4

        if(root.data ==1 or root.data ==0):
            root.data=2
            
        #case when both splits result in no examples     
        if(root.data==5):
            root.isAt=False
            root.left=None
            root.right=None
            if c1>c0:
                root.data=1
                root.count=c1
            else:
                root.data=0
                root.count=c0
            return root
        else:
            root.label=mxVar
        #we either need to build the tree in both directions, or just to the right or left, or just return the leaf node.
        if(root.data==2):
            root.left=viAlgo(df.where(df[mxVar]==0),identifier,attributes[:])
            root.right=viAlgo(df.where(df[mxVar]==1),identifier,attributes[:])
            return root
        elif(root.data==3):
            if c1>c0:
                root.left = dTree(data=1,a=False,c=c1)
            else:
                root.left = dTree(data=0,a=False,c=c0)
            root.right=viAlgo(df.where(df[mxVar]==1),identifier,attributes[:])
            return root
        elif(root.data==4):
            if c1>c0:
                root.right= dTree(data=1,a=False,c=c1)
            else:
                root.right = dTree(data=0,a=False,c=c0)
            root.left=viAlgo(df.where(df[mxVar]==0),identifier,attributes[:])
            return root
        else:
            return root

def printTree(root,dString):
    if(root.left.isAt is True):
        print(dString + root.label + " = 0 : ")
        printTree(root.left,dString+"|")
    else:
        print(dString + root.label + " = 0 : "+str(root.left.data))
    if(root.right.isAt is True):
        print(dString + root.label + " = 1 : ")
        printTree(root.right,dString+"|")
    else:
        print(dString + root.label + " = 1 : "+str(root.right.data))
def testTree(root,vals):
    if(root.right is not None and vals[root.label]==1):
        return testTree(root.right,vals)
    elif(root.left is not None and vals[root.label]==0):
        return testTree(root.left,vals)
    else:
        return root.data

def pruneTreeS(root):
    temp = root
    pruneTree(temp)
    return temp
    
def pruneTree(root):
    if(root.left.isAt is False and root.right.isAt is False):
        if(root.left.data==root.right.data):
            root.data = root.left.data
            root.count=root.left.count + root.right.count
            #this shouldnt happen
        else:
            c1 = root.right.count
            c0 = root.left.count
            if(c1>c0):
                root.data=root.right.data
                root.count=c1
            else:
                root.data=root.left.data
                root.count=c0
        root.isAt=False
        root.label=None
        root.right = None
        root.left =None
    elif(root.left.isAt is False):
        pruneTree(root.right)
    else:
        pruneTree(root.left)
        
    return  0    
try:
	trainingfile = sys.argv[1]
	validationfile = sys.argv[2]
	testfile=sys.argv[3]
	toPrint=sys.argv[4]
	prune=sys.argv[5]
except:
	print("Error passing in arguments, make sure to pass in all arguments")
	sys.exit(1)

print("args passed in successfully")
attributes,data=parseData(trainingfile)
attributes,vdf=parseData(validationfile)
attributes,tdf=parseData(testfile)
attributes=attributes[:-1] #removing class from attributes
i = 0
tot = vdf["Class"].count()
vcdf = vdf.drop("Class",axis=1)

y=id3Algo(data,"Class",attributes)
z=viAlgo(data,"Class",attributes)

correct = 0
correct2 = 0
correct3 = 0
correct4 = 0

while(i<tot):
    result=testTree(y,tdf.ix[i].to_dict())
    result2= testTree(z,tdf.ix[i].to_dict())
    result3=testTree(y,vdf.ix[i].to_dict())
    result4= testTree(z,vdf.ix[i].to_dict())
    #print i
    #print tdf["Class"].ix[i]
    #print y.label
    if(tdf["Class"].ix[i]==result):
        correct = correct + 1
    if(tdf["Class"].ix[i]==result2):
        correct2 = correct2 + 1
    if(vdf["Class"].ix[i]==result3):
        correct3 = correct3 + 1
    if(vdf["Class"].ix[i]==result4):
        correct4 = correct4 + 1
    i=i+1

print("id3Algorithm accuracy on test data before pruning: %f"%(float(correct)/tot))
print("viAlgorithm accuracy on test data before pruning: %f"%(float(correct2)/tot))
print("id3Algorithm accuracy on valid data before pruning: %f"%(float(correct3)/tot))
print("viAlgorithm accuracy on valid data before pruning: %f"%(float(correct4)/tot))
i=0
if(toPrint == "yes"):
        printTree(y,"")
        #printTree(z,"")
if(prune == "yes"):
    pruneTreeS(y)

    #TODO make this a loop and continue based on accuracy improvements, reset y to improved tree each time, and make testTree / accuracy a function call
