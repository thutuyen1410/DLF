import pandas as pd
from sklearn.datasets import load_svmlight_file

# Read pre-processed data from file
X, y = load_svmlight_file('diabetes_scale.txt')

# Convert into a dense numpy array
X_dense = X.toarray()

# Create a pandas DataFrame
data = pd.DataFrame(X_dense, columns=[f'Feature_{i}' for i in range(X_dense.shape[1])])

# Named 'Label' column for the class labels, and changed labels with value of -1 to 0
data['Label'] = y
data['Label'] = data['Label'].replace(-1,0)
print(data.head(10))

# Split the data into training, validation and testing sets:
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(data.drop('Label', axis=1), data['Label'], test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)

# Deep Learning imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Define some key parameters
hiddensizes = [16,32,64,128]
actfn = "relu"
# Optimiser and learning rate
opt_ins = keras.optimizers.Adam
learningrate = 0.001   
# Set size of batch and number of epochs
batch_size = 32
n_epochs = 50
 
# Build Perceptron model (using dense layers)
def perceptron(hiddensizes, actfn, optimizer, learningrate):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(8, activation = actfn))
    for n in hiddensizes:
        model.add(keras.layers.Dense(n, activation = actfn))
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer=optimizer(learning_rate=learningrate), metrics=["accuracy"])
    return model

def implement(hiddensizes, actfn, optimizer, learningrate, n_epochs, batch_size):
    model = perceptron(hiddensizes, actfn, optimizer, learningrate)
    history = model.fit(X_train, y_train, epochs=n_epochs, validation_data=(X_val, y_val))
    max_val_acc = np.max(history.history['val_accuracy'])
    return (max_val_acc, history, model)

def plot_history(history):
    plt.figure(figsize=(8,5))
    n = len(history.history['accuracy'])
    plt.plot(np.arange(0,n),history.history['accuracy'], color='orange')
    plt.plot(np.arange(0,n),history.history['loss'],'b')
    plt.plot(np.arange(0,n)+0.5,history.history['val_accuracy'],'r')  
    plt.plot(np.arange(0,n)+0.5,history.history['val_loss'],'g')
    plt.legend(['Train Acc','Train Loss','Val Acc','Val Loss'])
    plt.grid(True)
    plt.show() 

# Training
val_acc, history, model_trained = implement(hiddensizes, actfn, opt_ins, learningrate, n_epochs, batch_size)
print("Best accuracy: ", val_acc)

# To plot nice figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('figure', dpi=100)
import seaborn as sns; sns.set()
res = []
for n in [1,2,3,4]:
    valacc, history, discard = implement(hiddensizes[:n], actfn, opt_ins, learningrate, n_epochs, batch_size)
    res += [[n,valacc]]

print('Model perfomance of different number of hidden layers:')
print(res)
res=np.array(res)
plt.plot(res[:,0],res[:,1])
plt.plot(res[:,0],res[:,1],'o')
plt.title('Accuracy vs Layers')
plt.xlabel('Number of Layers')
plt.ylabel('Val Accuracy')

# Learning rate
res=[]
for lr in [1,0.1, 0.01,0.001,0.0001]:
    valacc, history, discard = implement(hiddensizes, actfn, opt_ins, lr, n_epochs, batch_size)
    plot_history(history)
    res += [[lr,valacc]]
print(res)

res=[]
optimizer_setup = [[keras.optimizers.SGD,0.001], [keras.optimizers.Adam,0.001],[keras.optimizers.RMSprop, 0.001], [keras.optimizers.Nadam, 0.001]]
for optimizer,lr in optimizer_setup :
    valacc, history, discard = implement(hiddensizes, actfn, optimizer, lr, n_epochs, batch_size)
    plot_history(history)
    res += [[valacc]]
print(res)

# final model
final_val_acc, final_history, final_model=implement(hiddensizes[:2], actfn, optimizer, 0.01, 100, batch_size)
scores = final_model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = final_model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_test_pred = (final_model.predict(X_test)>0.5).astype(int)
c_mat = confusion_matrix(y_test, y_test_pred, normalize='true')
ax = sns.heatmap(c_mat, annot=True, 
                 cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y_test_pred_probs = final_model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_test_pred_probs)

print("ROC AUC Score:", roc_auc)
 