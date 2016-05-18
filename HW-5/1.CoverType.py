# ### Cover Type
#optimal model: Boosted Trees
#Currently optimal depth: 15>14>13>12>11>10. I decided to keep using 15.
from pyspark import SparkContext
sc = SparkContext()
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel
path='/covtype/covtype.data'
inputRDD=sc.textFile(path)
Label=2.0
Data1=inputRDD.map(lambda line: [float(x) for x in line.split(',')]).map(lambda V:LabeledPoint(1.0 if V[-1]==Label else 0.0,V[:-1]))
(trainingData,testData)=Data1.randomSplit([0.7,0.3])
from time import time
errors={}
cfi = dict(zip(range(10,54),[2]*44))
for depth in [15]:
    start=time()
    model=GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo=cfi,loss="logLoss",maxDepth=depth,numIterations=10,learningRate=0.5,maxBins=20)
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda lp:lp.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
    #print depth,errors[depth],int(time()-start),'seconds'
    print depth,errors[depth]
