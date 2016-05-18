# ### Higgs data set
#currently optimal depth:
#Boosting: 5<6<7<8<9>10>11>12>13(9 with {'test': 0.27538660729411196, 'train': 0.2590240967456463} 19 seconds)
#Random Forest: 13 > 12>11>10 > 9>8
#du443n5ly optimal: GradientBoostedTrees, depth=9,lr=0.4,loss=logLoss
from pyspark import SparkContext
sc = SparkContext()
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel

path = '/HIGGS/HIGGS.csv'
inputRDD=sc.textFile(path)
Data=inputRDD.map(lambda line:[float(strip(x)) for x in line.split(',')]).map(lambda line:LabeledPoint(1 if line[0]>0.5 else 0,line[1:]))
Data1=Data.sample(False,0.1,seed=255).cache()
(trainingData,testData)=Data1.randomSplit([0.7,0.3],seed=255)
trainingData = trainingData.cache()
testData = testData.cache()

from time import time
errors={}
cfi = {}
depth = 9
lr = 0.3
lossfunc = "logLoss"
#stopER = 0.27498
stopER = 0.2745
testER = 1.0
attemption = 0
while testER>=stopER:
    start=time()
    model=GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo=cfi,loss="logLoss",maxDepth=depth,numIterations=10,learningRate=lr)
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in ['test','train']:  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda lp:lp.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
        if name=='test':
            if Err>=stopER:
                attemption+=1
                break
            else:
                testER = Err
print depth,errors[depth]
    #print "depth=",depth,", lr=",lr,", lossfunc=",lossfunc,errors[depth],int(time()-start),'seconds',', attemption=',attemption
