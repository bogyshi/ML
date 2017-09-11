import sys
import csv
import pandas as pd
import math

class dTree(object):
    def __init__(self,data=None,a=True):
        self.left = None
        self.right = None
        self.data = data
        self.label = None
        self.isAt = a

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
        #check to see if there are any attributes left to split on, return leaf node if nothing left after determing majority leader
        if(len(attributes)==0):
            root.isAt=False
            if c1>c0:
                root.data=1
            else:
                root.data=0
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
            if c1>c0:
                root.data=1
            else:
                root.data=0
            return root

        attributes.remove(mxVar) # this is an inplace operation

        #case where there is no example in training set when current split takes 0 branch
        if(df.where(df[mxVar]==0)[identifier].count()==0):
            root.data=3
            if c1>c0:
                root.left = dTree(data=1,a=False)
            else:
                root.left = dTree(data=0,a=False)
            
        #case where there is no example when we split with 1 branch
        if(df.where(df[mxVar]==1)[identifier].count()==0):
            root.data=4
            if c1>c0:
                root.right= dTree(data=1,a=False)
            else:
                root.right = dTree(data=0,a=False)
            
        #case when both splits result in no examples     
        if(root.left is not None and root.right is not None):
            root.isAt=False
            root.left=None
            root.right=None
            if c1>c0:
                root.data=1
            else:
                root.data=0
            return root
        else:
            root.label=mxVar
        #we either need to build the tree in both directions, or just to the right or left, or just return the leaf node.
        if(root.data is None):
            root.data = 2
            root.left=id3Algo(df.where(df[mxVar]==1),identifier,attributes[:])
            root.right=id3Algo(df.where(df[mxVar]==1),identifier,attributes[:])
            return root
        elif(root.data==3):
            root.right=id3Algo(df.where(df[mxVar]==1),identifier,attributes[:])
            return root
        elif(root.data==4):
            root.left=id3Algo(df.where(df[mxVar]==1),identifier,attributes[:])
            return root
        else:
            return root

def viAlgo(df,identifier,attributes):
        mx = 0.0
        root = dTree()
        if(len(attributes)==0):
                c1 = df.where(df[identifier]==1)[identifier].count()
                c0 = df.where(df[identifier]==0)[identifier].count()
                root.isAt=False
                if c1>c0:
                        root.data=1
                else:
                        root.data=0
                return root
                
        mxVar = attributes[0]
        for x in attributes[:]:
	        ent = infoGain(x,data.filter(items=[x,identifier]),identifier)
                if(ent>mx):
                        mx = ent
                        mxVar = x

        print("max info gain w/ entropy is %s with info gain of %f"%(mxVar,mx))
        if(mx==1):
                c1 = df.where(df[identifier]==1)[identifier].count()
                c0 = df.where(df[identifier]==0)[identifier].count()
                root.isAt=False
                if c1>c0:
                        root.data=1
                else:
                        root.data=0
                return root
        print mx
        attributes.remove(mxVar)
        l1=attributes[:]
 
        c1 = df.where(df[identifier]==1)[identifier].count()
        c0 = df.where(df[identifier]==0)[identifier].count()
        if(df.where(df[mxVar]==0)[identifier].count()==0):
                root.data=3
                if c1>c0:
                        root.left=1
                else:
                        root.left=0
        if(df.where(df[mxVar]==1)[identifier].count()==0):
                root.data=4
                if c1>c0:
                        root.right=1
                else:
                        root.right=0

        if(root.left is not None and root.right is not None):
                root.isAt=False
                root.data=5
        if(root.data != 5):
                root.label=mxVar
        else:
                root.label="noLabel"
        if(root.data is None):
                root.data = 2
        if(root.left is None):
                root.left=viAlgo(df.where(df[mxVar]==0),identifier,attributes[:])
        if(root.right is None):
                root.right=viAlgo(df.where(df[mxVar]==1),identifier,attributes[:])
        print root.data
        return root

def printTree(root,dString):
    if(root.data == 2):
        printTree(root.left,dString+" | ")
        printTree(root.right,dString+" | ")
    elif(root.data==3):
        print(dString + root.label + " = 0 : "+str(root.left.data))
        printTree(root.right,dString+" | ")
    elif(root.data==4):
        printTree(root.left,dString+" | ")
        print(dString + root.label + " = 1 : "+str(root.right.data))
    elif(root.data==5):
        print(dString + root.label + " = 0 : "+str(root.left.data))
        print(dString + root.label + " = 1 : "+str(root.right.data))
def testTree(root,vals):
    if(root.right is not None and vals[root.label]==1):
        return testTree(root.right,vals)
    elif(root.left is not None and vals[root.label]==0):
        return testTree(root.left,vals)
    else:
        return root.data

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
temp = y
print("Done with id3algo")
print y
#z=viAlgo(data,"Class",attributes)
result=0
correct = 0
while(i<tot):
    result=testTree(y,tdf.ix[i].to_dict())
    #print i
    #print tdf["Class"].ix[i]
    #print y.label
    if(tdf["Class"].ix[i]==result):
        correct = correct + 1
    i=i+1
    y = temp

print("id3Algorithm accuracy on test data before pruning: %f"%(float(correct)/tot))
i=0
correct=0
while(i<tot):
    result=testTree(y,data.ix[i].to_dict())
    #print data.ix[i].to_dict()
    #print data["Class"].ix[i]
    if(data["Class"].ix[i]==result):
        correct = correct + 1
    i=i+1
    y= temp
print("id3Algorithm accuracy on training data before pruning: %f"%(float(correct)/tot))

i=0
if(toPrint == "yes"):
        printTree(y,"")
        #printTree(z,"")
