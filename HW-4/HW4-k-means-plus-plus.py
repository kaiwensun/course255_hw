# -*- coding: utf-8 -*-
# Name: Kaiwen Sun
# Email: kas003@eng.ucsd.edu
# PID: A53091621
from pyspark import SparkContext
sc = SparkContext()
# coding: utf-8

# ## K-means++
# 
# In this notebook, we are going to implement [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) algorithm with multiple initial sets. The original k-means++ algorithm will just sample one set of initial centroid points and iterate until the result converges. The only difference in this implementation is that we will sample `RUNS` sets of initial centroid points and update them in parallel. The procedure will finish when all centroid sets are converged.

# In[1]:

### Definition of some global parameters.
K = 5  # Number of centroids
RUNS = 25  # Number of K-means runs that are executed in parallel. Equivalently, number of sets of initial points
RANDOM_SEED = 60295531
converge_dist = 0.1 # The K-means algorithm is terminated when the change in the location 
                    # of the centroids is smaller than 0.1


# In[28]:

import numpy as np
import pickle
import sys
from numpy.linalg import norm
#from matplotlib import pyplot as plt


def print_log(s):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def parse_data(row):
    '''
    Parse each pandas row into a tuple of (station_name, feature_vec),
    where feature_vec is the concatenation of the projection vectors
    of TAVG, TRANGE, and SNWD.
    '''
    return (row[0],
            np.concatenate([row[1], row[2], row[3]]))


def compute_entropy(d):
    '''
    Compute the entropy given the frequency vector `d`
    '''
    d = np.array(d)
    d = 1.0 * d / d.sum()
    return -np.sum(d * np.log2(d))


def choice(p):
    '''
    Generates a random sample from [0, len(p)),
    where p[i] is the probability associated with i. 
    '''
    random = np.random.random()
    r = 0.0
    for idx in range(len(p)):
        r = r + p[idx]
        if r > random:
            return idx
    assert(False)


def kmeans_init(rdd, K, RUNS, seed):
    '''
    Select `RUNS` sets of initial points for `K`-means++
    '''
    # the `centers` variable is what we want to return
    n_data = rdd.count()
    shape = rdd.take(1)[0][1].shape[0]
    centers = np.zeros((RUNS, K, shape))

    def update_dist(vec, dist, k):
        new_dist = norm(vec - centers[:, k], axis=1)**2
        return np.min([dist, new_dist], axis=0)


    # The second element `dist` in the tuple below is the closest distance from
    # each data point to the selected points in the initial set, where `dist[i]`
    # is the closest distance to the points in the i-th initial set.
    data = rdd.map(lambda p: (p, [np.inf] * RUNS))               .cache()

    # Collect the feature vectors of all data points beforehand, might be
    # useful in the following for-loop
    local_data = rdd.map(lambda (name, vec): vec).collect()

    # Randomly select the first point for every run of k-means++,
    # i.e. randomly select `RUNS` points and add it to the `centers` variable
    sample = [local_data[k] for k in np.random.randint(0, len(local_data), RUNS)]
    centers[:, 0] = sample

    for idx in range(K - 1):
        data1 = data
        #找出来每个vec距离到idx个种子点中最近的那个种子点的距离
        for k in range(idx+1):
            data1 = data1.map(lambda ((name,vec),dist):((name,vec),update_dist(vec,dist,k))).cache()
        
        #内循环结束后，每一行对应一个数据点，是：((name,vec),[在25个宇宙里到最近的黑洞的距离])
        
        #把距离normalize成和为1的概率
        #data2里有n行，每行对应一个星球（数据点）。每行的内容是一个长度为25的向量，表示这个星球在各个宇宙中距离最近黑洞的距离（的平方）。
        data2 = data1.map(lambda ((name,vec),dist):dist).cache()
        #summation是一个长度为25的向量，表示每个宇宙中dist的和
        summation = data2.reduce(lambda distsq1,distsq2:distsq1+distsq2)
        
        #data3是个n乘25的矩阵，每列对应一个宇宙，每列的和都是1，表示一个dist归一化了的宇宙
        data3 = np.array(data2.map(lambda distsq:distsq/summation).collect())
        
        #index 是一个向量，长度为25，表示每个宇宙中新变成（第k个）黑洞的星球的index
        index = np.apply_along_axis(func1d=choice,axis=0,arr=data3)
        
        #根据星球坐标表(local_data，各个宇宙共享)，把各个宇宙中新黑洞的index翻译成新黑洞的坐标
        centers[:,idx+1]=np.array([local_data[i] for i in index])
        
        ##############################################################################
        # Insert your code here:
        ##############################################################################
        # In each iteration, you nee(da,di) (da,update_dist(local_data,di,idx))to select one point for each set
        # of initial points (so select `RUNS` points in total).
        # For each data point x, let D_i(x) be the distance between x and
        # the nearest center that has already been added to the i-th set.
        # Choose a new data point for i-th set using a weighted probability
        # where point x is chosen with probability proportional to D_i(x)^2
        ##############################################################################
        #data = rdd.map(lambda p: (p, [np.inf] * RUNS)) \
        #      .cache()

    return centers


def get_closest(p, centers):
    '''
    Return the indices the nearest centroids of `p`.
    `centers` contains sets of centroids, where `centers[i]` is
    the i-th set of centroids.
    '''
    best = [0] * len(centers)
    closest = [np.inf] * len(centers)
    for idx in range(len(centers)):
        for j in range(len(centers[0])):
            temp_dist = norm(p - centers[idx][j])
            if temp_dist < closest[idx]:
                closest[idx] = temp_dist
                best[idx] = j
    return best


def kmeans(rdd, K, RUNS, converge_dist, seed):
    '''
    Run K-means++ algorithm on `rdd`, where `RUNS` is the number of
    initial sets to use.
    '''
    #k_points.shape = (RUNS,K,9) = (25,5,9)。25个平行宇宙，每个宇宙里选出了5个黑洞
    k_points = kmeans_init(rdd, K, RUNS, seed)
    
    #print "k_points=",k_points
    #print "k_points.shape=",k_points.shape
    
    print_log("Initialized.")
    temp_dist = 1.0

    iters = 0
    st = time.time()
    
    #rdd的每一行是(name,vec),其中vec是星球的坐标（长度为9）
    rdd = rdd.cache()
    shape = rdd.take(1)[0][1].shape[0]
    
    while temp_dist > converge_dist:
        new_points = np.zeros((RUNS, K, shape))
        
        #data1有n行，每行对应一个星球，每一行是(星球坐标，星球在25个宇宙里所附属于的黑洞id)
        #data1 = rdd.map(lambda (name,vec):(vec,get_closest(vec,k_points)))
        
        #data2有n行，每行对应一个星球，每一行是(星球坐标，[(0,该星球在第0个宇宙里所属的黑洞id),...,(24,该星球在第24个宇宙里所属的黑洞id)])
        #data2 = data1.map(lambda (vec,ids):(vec,zip(range(25),ids)))
        
        #data3的每一项是一个key-value pair。key是(宇宙id,黑洞id)，value是(一个附属于这个宇宙中的这个黑洞上的星球的坐标,1)
        #data3 = data2.flatMap(lambda (vec,lst_univid_bhid):[(univid_bhid,(vec,1)) for univid_bhid in lst_univid_bhid])
        
        #compress computations above to one line
        data3 = rdd.flatMap(lambda (name,vec):[((univid,bhid),(vec,1)) for (univid,bhid) in zip(range(25),get_closest(vec,k_points))])
        
        #daat4一共有25*5项，每一项是一个key-value pair。key是(宇宙id,黑洞id)，value是(附属于这个宇宙中的这个黑洞上的星球距离黑洞的距离之和,附属的星球的个数)
        data4 = data3.reduceByKey(lambda (vec1,cnt1),(vec2,cnt2):(vec1+vec2,cnt1+cnt2))
        
        #data5一共有25×5项，每一项是一个key-value pair。key是(宇宙id，黑洞id)，value是这个宇宙里的这个黑洞的新坐标
        data5 = data4.map(lambda ((univid,bhid),(vecsum,cnt)):((univid,bhid),vecsum/cnt)).collect()

        for ((univid,bhid),newvec) in data5:
            new_points[univid][bhid]=newvec
        
        
        ##############################################################################
        # INSERT YOUR CODE HERE
        ##############################################################################
        
        # Update all `RUNS` sets of centroids using standard k-means algorithm
        # Outline:
        #   - For each point x, select its nearest centroid in i-th centroids set
        #   - Average all points that are assigned to the same centroid
        #   - Update the centroid with the average of all points that are assigned to it
        
        # Insert your code here

        # You can modify this statement as long as `temp_dist` equals to
        # max( sum( l2_norm of the movement of j-th centroid in each centroids set ))
        ##############################################################################

        temp_dist = np.max([
                np.sum([norm(k_points[idx][j] - new_points[(idx, j)]) for j in range(K)])
                    for idx in range(RUNS)])

        iters = iters + 1
        if iters % 5 == 0:
            print_log("Iteration %d max shift: %.2f (time: %.2f)" %
                      (iters, temp_dist, time.time() - st))
            st = time.time()

        # update old centroids
        # You modify this for-loop to meet your need
        #for ((idx, j), p) in new_points.items():
        #    k_points[idx][j] = p
        k_points = new_points
        
        #print_log("iters=%d"%(iters))

    return k_points


# In[3]:

## Read data
data = pickle.load(open("../Data/Weather/stations_projections.pickle", "rb"))
rdd = sc.parallelize([parse_data(row[1]) for row in data.iterrows()])
rdd.take(1)


# In[29]:

# main code

import time

st = time.time()

np.random.seed(RANDOM_SEED)
centroids = kmeans(rdd, K, RUNS, converge_dist, np.random.randint(1000))
group = rdd.mapValues(lambda p: get_closest(p, centroids))            .collect()

print "Time takes to converge:", time.time() - st


# ## Verify your results
# Verify your results by computing the objective function of the k-means clustering problem.

# In[8]:

def get_cost(rdd, centers):
    '''
    Compute the square of l2 norm from each data point in `rdd`
    to the centroids in `centers`
    '''
    def _get_cost(p, centers):
        best = [0] * len(centers)
        closest = [np.inf] * len(centers)
        for idx in range(len(centers)):
            for j in range(len(centers[0])):
                temp_dist = norm(p - centers[idx][j])
                if temp_dist < closest[idx]:
                    closest[idx] = temp_dist
                    best[idx] = j
        return np.array(closest)**2
    
    cost = rdd.map(lambda (name, v): _get_cost(v, centroids)).collect()
    return np.array(cost).sum(axis=0)

cost = get_cost(rdd, centroids)


# In[25]:

log2 = np.log2

print log2(np.max(cost)), log2(np.min(cost)), log2(np.mean(cost))


# ## Plot the increase of entropy after multiple runs of k-means++

# In[30]:

entropy = []

for i in range(RUNS):
    count = {}
    for g, sig in group:
        _s = ','.join(map(str, sig[:(i + 1)]))
        count[_s] = count.get(_s, 0) + 1
    entropy.append(compute_entropy(count.values()))


# **Note:** Remove this cell before submitting to PyBolt (PyBolt does not fully support matplotlib)

# In[31]:

#get_ipython().magic(u'matplotlib inline')

#plt.xlabel("Iteration")
#plt.ylabel("Entropy")
#plt.plot(range(1, RUNS + 1), entropy)
2**entropy[-1]


# ## Print the final results

# In[32]:

print 'entropy=',entropy
best = np.argmin(cost)
print 'best_centers=',list(centroids[best])


# In[ ]:



