import sys
import csv
import pandas as pd
import math
from copy import deepcopy

class dTree(object):
    def __init__(self,datar=None,a=True,c=0):
        self.left = None
        self.right = None
        self.data = datar
        self.label = None
        self.isAt = a
        self.count = c
    def removeNodes(self,count,data):
        self.isAt=False
        self.data=data
        self.count=count
        self.left=None
        self.right=None
    def returnNodes(self,temp1,temp2,temp3):
        self.label=temp1
        self.right=temp2
        self.left=temp3
        self.isAt=True
def parseData(filename):
    attributes=[]
    data=[]
    f = open(filename,'r')
    attributes=f.readline().split(",")
    attributes[-1]=attributes[-1][:-1] #removing newline character
    df = pd.read_csv(filename)
    with open(filename,'r') as csvFile:
        reader = csv.reader(csvFile,delimiter=',')
        for x in reader:
            data.append(x)
    return attributes,df

def infoGainS(df,identifier,attributes):

    net1 = df.where(df[identifier]==1)[identifier].count()
    net0 = df.where(df[identifier]==0)[identifier].count()
    totnet = net1+net0
    pnet1=float(net1)/totnet
    pnet0=float(net0)/totnet
    if(pnet1>0):
        x1=-1*pnet1*math.log(pnet1,2)
    else:
        x1=0
    if(pnet0>0):
        y1=-1*pnet0*math.log(pnet0,2)
    else:
        y1=0
    mx = 0
    mxVar=""
    resultnet = x1+y1
    for x in attributes[:]:
        h = entropies(x,df.filter(items=[x,identifier]),identifier,totnet)
        ent = resultnet+h
        if(ent>mx):
            mx = ent
            mxVar = x
    #print (mxVar)
    return mx, mxVar

def entropies(target,df,identifier,totnet):

    #now entropy of each variable in attribute TARGET
    first = df.where(df[target]==1) # only where attribute is 1
    second = df.where(df[target]==0) # only where attribute is 0
    pos1 = first.where(df[identifier]==1)[identifier].count() #where attribute is 1 and class is 1
    pos0 = first.where(df[identifier]==0)[identifier].count() # attribute 1 and class is 0
    nul1 = second.where(df[identifier]==1)[identifier].count() # attribute 0 and class 1
    nul0 = second.where(df[identifier]==0)[identifier].count() #attribute 0 and class 0
    tot1 = pos1+pos0
    tot2 = nul1+nul0
    if(tot1>0):
        p1=(float(pos1)/tot1)
        p2=(float(pos0)/tot1)
    else:
        p1=0
        p2=0
    if(tot2>0):
        n1=(float(nul1)/tot2)
        n2=(float(nul0)/tot2)
    else:
        n1=0
        n2=0
    if(p1 == 0 and p2==0):
        resultp=0
    elif(p1==0):
        resultp = -1*p2*math.log(p2,2)
    elif(p2==0):
        resultp = -1*p1*math.log(p1,2)
    else:
        resultp=-1*p1*math.log(p1,2) + -1*p2*math.log(p2,2)

    if(n1==0 and n2==0):
        resultn=0
    elif(n1==0):
        resultn = -1*n2*math.log(n2,2)
    elif(n2==0):
        resultn = -1*n1*math.log(n1,2)
    else:
        resultn = -1*n1*math.log(n1,2) + -1*n2*math.log(n2,2)
    c1=float(tot1)/totnet
    c0=float(tot2)/totnet
    
    x=  -1*(c1)*resultp + -1*(c0)*resultn #info gain
    
    return x

def VIS(df,identifier,attributes):
    #calc entire set vi
    net1 = df.where(df[identifier]==1)[identifier].count()
    net0 = df.where(df[identifier]==0)[identifier].count()
    totnet = net1+net0
    if(net1!=0 and net0!=0):
        pnet1=float(net1)/totnet
        pnet0=float(net0)/totnet
        vinet = (net1*net0)/(float(totnet)*totnet)
    else:
        vinet = 0

    mx=0.0
    mxVar=""
    for x in attributes[:]:
        h = vimpur(x,df.filter(items=[x,identifier]),identifier,totnet)
        ent = vinet+h
        if(ent>mx):
            mx = ent
            mxVar = x
    return mx, mxVar
def vimpur(target,df,identifier,totnet):

    first = df.where(df[target]==1) # only where attribute is 1
    second = df.where(df[target]==0) # only where attribute is 0
    pos1 = first.where(df[identifier]==1)[identifier].count() #where attribute is 1 and class is 1
    pos0 = first.where(df[identifier]==0)[identifier].count() # attribute 1 and class is 0
    nul1 = second.where(df[identifier]==1)[identifier].count() # attribute 0 and class 1
    nul0 = second.where(df[identifier]==0)[identifier].count() #attribute 0 and class 0

    tot1 = pos1+pos0
    tot2 = nul1+nul0
    if(tot1!=0 and tot2!=0):

        p1=(float(tot1)/totnet)
        n1=(float(tot2)/totnet)
        resultp = pos1*pos0/(float(tot1)*tot1)
        resultn = nul1*nul0/(float(tot2)*tot2)

        return -1*p1*resultp + -1*n1*resultn
    elif(tot1==0):
        n1=(float(tot2)/totnet)
        resultn = nul1*nul0/(float(tot2)*tot2)

        return -1*n1*resultn
    elif(tot2==0):
        p1=(float(tot1)/totnet)
        resultp = pos1*pos0/(float(tot1)*tot1)

        return -1*p1*resultp
    else:
        return 0

def id3Algo(df,identifier,attributes,dt,ct,IDorVI):
    df=df.dropna()
    root = dTree()
    c1 = df.where(df[identifier]==1)[identifier].count()
    c0 = df.where(df[identifier]==0)[identifier].count()
    if(c1==0 and c0 == 0):
        #print("no examples")
        root.data = dt
        root.count = ct
        root.isAt=False
        return root
    elif(c1==0):
        root.data = 0
        root.count = c0
        root.isAt=False
        return root
    elif(c0==0):
        root.data = 1
        root.count = c1
        root.isAt=False
        return root
    elif c1>c0:
        root.data=1
        root.count=c1
    else:
        root.data=0
        root.count=c0

        #check to see if there are any attributes left to split on, return leaf node if nothing left after determing majority leader
    if(len(attributes)==0):
        root.isAt=False
        #print("no more attributes")
        return root

    #determine variable of greatest info gain
    if(IDorVI is True):
        mx,mxVar = infoGainS(df,identifier,attributes[:])
    else:
        mx, mxVar= VIS(df,identifier,attributes[:])
    if(mx==0):
            root.isAt = False
            return root
    attributes.remove(mxVar) # this is an inplace operation

    root.label=mxVar
    root.left=id3Algo(df.where(df[mxVar]==0),identifier,attributes[:],root.data,root.count,IDorVI)
    root.right=id3Algo(df.where(df[mxVar]==1),identifier,attributes[:],root.data,root.count,IDorVI)
    return root

def printTree(root,dString):
    if(root.left.isAt is True):
        print(dString + root.label + " = 0 :")
        printTree(root.left,dString+"| ")
    else:
        print(dString + root.label + " = 0 :"+str(root.left.data))
    if(root.right.isAt is True):
        print(dString + root.label + " = 1 :")
        printTree(root.right,dString+"| ")
    else:
        print(dString + root.label + " = 1 : "+str(root.right.data))

def testTree(rooter,valss):
    #print (rooter.label + " heading to the "+ str(valss[rooter.label]))
    if(rooter is None):
        print ("wtf")
    if((rooter.right.isAt is True) and valss[rooter.label]==1):
        return testTree(rooter.right,valss)
    elif((rooter.right.isAt is False) and valss[rooter.label]==1):
        return rooter.right.data
    elif((rooter.left.isAt is True) and valss[rooter.label]==0):
        return testTree(rooter.left,valss)
    elif((rooter.left.isAt is False) and valss[rooter.label]==0):
        return rooter.left.data
    else:
        return rooter.data
def accuracyTest(head,dftest):
    ij=0
    tottest = dftest["Class"].count()
    correcttest=0
    temptest = (head)
    while(ij<tottest):
        tt = dftest.ix[ij].to_dict()
        result=testTree(temptest,tt)
        if(result!=1 and result!=0):
            print (result)
        if(int(tt["Class"]) == int(result)):
            correcttest = correcttest + 1
        #else:
            #print ("incorrect! result is %d while actual is %d at row %d"%(result,tt["Class"],ij) + "with path" + str(tt))
        ij=ij+1
    return float(correcttest)/ij

def pruneTreeS(root2,dfprune):
    #print(root2)
    stack=[root2]
    store=[]
    i = 0
    noImprov=True
    while (stack):
        root=stack.pop()
        if(root.left.isAt is True and root.right.isAt is True): # go
            stack.append(root.left)
            stack.append(root.right)
        elif(root.left.isAt is False and root.right.isAt is False):
            ogAc = accuracyTest(root2,dfprune)
            if(root.left.data==root.right.data):
                d = root.left.data
                c=root.left.count + root.right.count
            else:
                c1 = root.right.count
                c0 = root.left.count
                if(c1>c0):
                    d=root.right.data
                    c=c1
                else:
                    d=root.left.data
                    c=c0
            temp1=root.label
            temp2=root.right
            temp3=root.left
            root.removeNodes(c,d)
            postAc = accuracyTest(root2,dfprune)
            if(ogAc<postAc):
                hol=2
                noImprov=False
            else:
                root.returnNodes(temp1,temp2,temp3)

        elif(root.left.isAt is False):
            stack.append(root.right)
        else:
            stack.append(root.left)
        i=i+1
    return noImprov

def pruneTreeSR(root2,dfprune):
    tempr = root2
    pruneTreeR(root2,tempr,dfprune)
    return True

def pruneTreeR(root2,root,dfprune):
    if(root.left.isAt is True and root.right.isAt is True):
        pruneTreeR(root2,root.left,dfprune)
        pruneTreeR(root2,root.right,dfprune)
    elif(root.left.isAt is True):
        pruneTreeR(root2,root.left,dfprune)
    elif(root.right.isAt is True):
        pruneTreeR(root2,root.right,dfprune)
    if(root.left.isAt is False and root.right.isAt is False):
        ogAc = accuracyTest(root2,dfprune)
        if(root.left.data==root.right.data):
            d = root.left.data
            c=root.left.count + root.right.count
        else:
            c1 = root.right.count
            c0 = root.left.count
            if(c1>c0):
                d=root.right.data
                c=c1
            else:
                d=root.left.data
                c=c0
        temp1=root.label
        temp2=root.right
        temp3=root.left
        root.removeNodes(c,d)
        postAc = accuracyTest(root2,dfprune)
        if(ogAc<postAc):
            hol=2
            noImprov=False
        else:
            root.returnNodes(temp1,temp2,temp3)
        

try:
	trainingfile = sys.argv[1]
	validationfile = sys.argv[2]
	testfile=sys.argv[3]
	toPrint=sys.argv[4]
	prune=sys.argv[5]
except:
	print("Error passing in arguments, make sure to pass in all arguments")
	sys.exit(1)

def main():
    #print("args passed in successfully")
    attributes,dftrain=parseData(trainingfile)
    attributes=attributes[:-1] #removing class from attributes
    attributes2=attributes
    iGT=id3Algo(dftrain,"Class",attributes,0,600,True)
    vIT=id3Algo(dftrain,"Class",attributes2,0,600,False)
    #print("done with alg")
    idc2,validdf=parseData(validationfile)
    idc1,testdf=parseData(testfile)
    i = 0
    acid3 = accuracyTest(iGT,validdf)

    print("InfoGainAlgorithm accuracy on valid data before pruning: %f"%(acid3))
    print("InfoGainAlgorithm accuracy on test data before pruning: %f"%(accuracyTest(iGT,testdf)))
    print("VIAlgorithm accuracy on valid data before pruning: %f"%(accuracyTest(vIT,validdf)))
    print("VIAlgorithm accuracy on test data before pruning: %f"%(accuracyTest(vIT,testdf)))
 
    if(toPrint == "yes"):
            printTree(y,"")
            printTree(z,"")
    if(prune == "yes"):
        while True:
            x=pruneTreeSR(iGT,validdf)
            if(x is True):
                break
        while True:
            x=pruneTreeSR(vIT,validdf)
            if(x is True):
                break
        print("InfoGainAlgorithm accuracy on valid data after pruning: %f"%(accuracyTest(iGT,validdf)))
        print("viAlgorithm accuracy on valid data after pruning: %f"%(accuracyTest(vIT,validdf)))
        print("InfoGainAlgorithm accuracy on test data after pruning: %f"%(accuracyTest(iGT,testdf)))
        print("viAlgorithm accuracy on test data after pruning: %f"%(accuracyTest(vIT,testdf)))
        if(toPrint=="yes"):
            printTree(y,"")
            printTree(z,"")
main()
    

