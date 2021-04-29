#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


# load thyroid database
data = pd.read_csv('new-thyroid.csv')
data.sample(n=5)


# In[23]:


data.groupby('Class').Class.count()


# In[3]:


data.describe()


# In[4]:


#Normalize the data
data.Class = data.Class.astype('str')
data['Class'].replace(['1', '2', '3'], [0, 1, 2], inplace=True)
df_norm = data[['1', '2', '3', '4', '5']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)


# In[5]:


df_norm.describe()


# In[6]:


#Convert the Class labels to indexes for use with neural network.
target = data[['Class']]
target.sample(n=5)


# In[7]:


df = pd.concat([df_norm, target], axis=1)
df.sample(n=5)


# In[8]:


#Mark some of the data for testing purpose.
#We'll test our network on unseen data.
train_test_per = 90/100.0
df['train'] = np.random.rand(len(df)) < train_test_per
df.sample(n=5)


# In[9]:


#Separate train and test Data
train = df[df.train == 1]
train = train.drop('train', axis=1).sample(frac=1)
# train.sample(n=5)
train


# In[10]:


test = df[df.train == 0]
test = test.drop('train', axis=1)
# test.sample(n=5)
test


# In[11]:


X = train.values[:,:5]
X[:5]


# In[12]:


targets = [[1,0,0], [0,1,0], [0,0,1]]
y = np.array([targets[int(x)] for x in train.values[:,5:6]])
y[:5]


# In[13]:


#Create backpropagating neural network
#Create 3 layers: Input, hidden and Output.
#Inputs = length and widths of the species
#Output = 3 values, each one indicating a species. ie Values 1, 0, 0 for the output indicates Iris-setosa
#w1 is a matrices of weight connecting Input and the hidden layer. Each node in input layer connects to each node in the hidden layer.
#Weight are randomized between -1 and 1.

num_inputs = len(X[0])
hidden_layer_neurons = 5
np.random.seed(4)
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
w1


# In[14]:


#w2 are the weights of connections between hidden layer and output layer.
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
w2


# In[15]:


# taken from> https://gist.github.com/craffel/2d727968c3aaebd10359
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)


# In[16]:



#A Graphical representations of our network will be something like below
#The first set of 5 nodes is the input.
#The second set of 5 nodes is the hidden layer.
#The last set of 3 nodes is the output layer.
#All the nodes of a layer are fully connected to all nodes of the next layer.

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [5, 5, 3])


# In[17]:


# sigmoid function representation
_x = np.linspace( -5, 5, 50 )
_y = 1 / ( 1 + np.exp( -_x ) )
plt.plot( _x, _y )


# In[18]:


learning_rate = 0.1 # slowly update the network
for epoch in range(1000):
    l1 = 1/(1 + np.exp(-(np.dot(X, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    er = (abs(y - l2)).mean()
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += X.T.dot(l1_delta) * learning_rate
print('Error:', er)


# In[19]:


#Test the network for accuracy.
#Run the network with the updated weights from training.

X = test.values[:,:5]
y = np.array([targets[int(x)] for x in test.values[:,5:6]])

l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

np.round(l2,3)


# In[20]:


yp = np.argmax(l2, axis=1) # prediction
sample_test = np.argmax(y, axis=1)
res = yp == sample_test
correct = np.sum(res)/len(res)

testres = test[['Class']].replace([0, 1, 2], ['1', '2', '3'])

testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0, 1, 2], ['1', '2', '3'])

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')


# In[21]:


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class_names = ['1', '2', '3']

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [0, 1, 2]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[22]:


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(sample_test, yp, classes=class_names, 
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(sample_test, yp, classes=class_names , normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:




