from numpy import exp, array, random, dot
import numpy as np
import pandas as pd
import sys
# to visualize only
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import FeatureTuning
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


class NeuralNetwork():
	def __init__(self):
		#we intialize random weights with default values. Can be changed later.
		#number of nodes in hidden layer are represented by the second value in tuple. 
		self.hiddenlayer_size=(9000,4)
		self.hiddenlayer2_size=(4,60)
		#number of nodes in output layer are 40.
		self.outputLayer_size=(60,40)
		self.epoch=100
		self.l_rate=0.01
		self.initializeRandomWeights()

	def initializeRandomWeights(self):
		# Seed the random number generator, so it generates the same numbers
		# every time the program runs.
		#random.seed(1)
		
		# We assign random weights within the range -1 to 1
		self.synaptic_weights_H1 = 2*random.random(self.hiddenlayer_size)+1
		self.synaptic_weights_H2 = 2*random.random(self.hiddenlayer2_size)+1
		self.synaptic_weights_O = 2*random.random(self.outputLayer_size)+1
		
	# The Sigmoid function, which describes an S shaped curve.
	# We pass the weighted sum of the inputs through this function to
	# normalise them between 0 and 1.
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))
		
	def __leakyRelu(self,x):
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				if(x[i][j])<0:
					x[i][j]=0.1*x[i][j]
					#x[i][j]=0
		
		return x
	
	def __relu_derivative(self,x):
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				if(x[i][j])>0:
					x[i][j]=1
				else:
					x[i][j]=0
		
		return x

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	# It indicates how confident we are about the existing weight.
	def __sigmoid_derivative(self, x):
		return x * (1 - x)
		
	def tanh(self,x):
		return np.tanh(x)

	def __tanh_derivative(self,x):
		return 1.0 - x**2

	# We train the neural network through a process of trial and error.
	# Adjusting the weights each time.
	def train(self, training_set_inputs, trainY, X_test,y_test, number_of_training_iterations):
		J_history = np.zeros(shape=(number_of_training_iterations, 1))
		J_history_test = np.zeros(shape=(number_of_training_iterations, 1))
		for iteration in xrange(number_of_training_iterations):
			print "iteration", iteration
			#random.seed(iteration)
			#self.initializeRandomWeights()
			# Pass the training set through our neural network (a single neuron). 
			H1_output = self.neuron_tanH(training_set_inputs,self.synaptic_weights_H1)
			H2_output=self.neuron_tanH(H1_output,self.synaptic_weights_H2)
			O_output = self.neuron_tanH(H2_output,self.synaptic_weights_O)
			
			# Calculate the error (The difference between the desired output
			# and the predicted output)
			error_in_O = trainY - O_output
			J_history[iteration, 0] =np.mean(error_in_O)
			# Multiply the error by the input and again by the gradient of the Sigmoid curve.
			# This means less confident weights are adjusted more.
			# This means inputs, which are zero, do not cause changes to the weights.
			delta_O=error_in_O * self.__tanh_derivative(O_output)
			adjustment_in_O = self.l_rate* dot(H2_output.T,delta_O)
			
			#back- propagate to the hidden layer 

			#delta_weighted=np.mean(delta_O,axis=1).reshape(-1,1)
			delta_H2=dot(delta_O,self.synaptic_weights_O.T)
			#error_in_H=dot(delta_weighted,np.mean(self.synaptic_weights_H1,axis=0).reshape(1,-1))
			error_in_H2=self.l_rate* self.__tanh_derivative(H2_output)*delta_H2
			
			adjustment_in_H2 = self.l_rate*dot(H1_output.T,error_in_H2)
			
			#delta_weighted=np.mean(delta_O,axis=1).reshape(-1,1)
			delta_H1=dot(delta_H2,self.synaptic_weights_H2.T)
			#error_in_H=dot(delta_weighted,np.mean(self.synaptic_weights_H1,axis=0).reshape(1,-1))
			error_in_H1=self.l_rate* self.__relu_derivative(H1_output)*delta_H1
			
			adjustment_in_H1 = self.l_rate*dot(training_set_inputs.T,error_in_H1)
			
			# Adjust the weights.
			self.synaptic_weights_H2 += adjustment_in_H2
			
			self.synaptic_weights_H1+=adjustment_in_H1
			 
			#update weights in output layer
			self.synaptic_weights_O+=adjustment_in_O
			J_history_test[iteration, 0]=self.predict(X_test,y_test)
		return J_history,J_history_test
			
			
	# The neuron using sigmoid as activation function.
	def neuron_relu(self, inputs,weights):
		# Pass inputs through our neural network (our single neuron).
		return self.__leakyRelu(dot(inputs, weights))
		
	# The neuron using sigmoid as activation function.
	def neuron_tanH(self, inputs,weights):
		# Pass inputs through our neural network (our single neuron).
		return self.tanh(dot(inputs, weights))
		
	# The neuron using sigmoid as activation function.
	def neuron(self, inputs,weights):
		# Pass inputs through our neural network (our single neuron).
		return self.__sigmoid(dot(inputs, weights))
		
	def predict(self, testX,testY):
		print "The Result is:"
		#pass input to hidden layer
		H1_output=self.neuron_tanH(testX,self.synaptic_weights_H1)
		
		H2_output=self.neuron_tanH(H1_output,self.synaptic_weights_H2)
		#pass output of hidden layer to the output layer
		predictedValues=self.neuron_tanH(H2_output,self.synaptic_weights_O)
		#get dataframe to calculate the maximum value 
		dfP=pd.DataFrame(predictedValues)
		dfP['Max'] = dfP.idxmax(axis=1)
		predicted=dfP['Max'].values
		predicted=predicted.reshape(-1,1)
		if testY is None:
			print "testy" 
		else :
			dfA=pd.DataFrame(testY)
			dfA['Max'] = dfA.idxmax(axis=1)
			actual=dfA['Max'].values
			actual=actual.reshape(-1,1)
			print predicted,actual
			print accuracy_score(actual,predicted)
			print "Classification score is :", classification_report(actual,predicted)
			error=actual-predicted
		return np.mean(error)
		
	def standardizeTheData(self,X):
		
		#X= X/ np.max(np.abs(X)) 
		X=(X-np.mean(X))/(np.max(np.abs(X))-np.min(np.abs(X)))
		#X=(X-np.mean(X))/np.std(X)*255.0
		#X=(X-np.mean(X))/256.0
		
		return X

		
	def loadDataAndInitProcessing(self):
		train_fileX=sys.argv[1]
		train_fileY=sys.argv[2]
		trainX = np.load(train_fileX) # this should have shape (26344, 3, 64, 64)
		trainY = np.load(train_fileY)
		print "Need to augment data and get balanced data set? enter y/n->"
		augment=raw_input()
		if(augment=='y'):
			FeatureTuning.threshold=1000
			trainX,trainY=FeatureTuning.UnderSampleTheData(trainX,trainY)
			trainX,trainY=FeatureTuning.AugmentTheData(trainX,trainY)
		
		#testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)
		
		trainY=trainY.reshape(-1,1)
		#returns an array of shape (n,40)
		tunedLabels=FeatureTuning.EncodeClassLabels(trainY)
		#print trainY.shape
		#FeatureTuning.edgeDetection=True
		FeatureTuning.cropping=True
		FeatureTuning.HOG=True
		training_set_inputs=FeatureTuning.Transform3dTo1D(trainX)
		training_set_inputs=self.standardizeTheData(training_set_inputs)
		#shape=training_set_inputs.shape[1]
		#print shape
		#self.hiddenlayer_size=(shape,3);
		#self.initializeRandomWeights()
		np.random.shuffle(training_set_inputs)
		np.random.shuffle(tunedLabels)
		self.ApplyKFoldCrossValidation(training_set_inputs,tunedLabels)
		
	def ApplyKFoldCrossValidation(self,trainX,trainY):
		#testing_set_inputs=feature_tuning.Transform3dTo1D(testX)
		totalImages=len(trainX)
		k_fold = KFold(n=totalImages, n_folds=3)
		#for train_indices, test_indices in k_fold:
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainX, trainY, test_size=0.3, random_state=0)
			# Train the neural network using a training set.
			# Do it 100 times and make small adjustments each time.
			#self.train(trainX[train_indices], trainY[train_indices], self.epoch)
			#self.predict(trainX[test_indices],trainY[test_indices])
		J_history,J_history_test=self.train(X_train, y_train,X_test,y_test, self.epoch)
		#J_history_test=self.predict(X_test,y_test)
		#plt.plot(np.arange(iterations), J_history,'r'),np.arange(iterations), J_history_test,'b')
		train_error=plt.plot(np.arange(self.epoch), J_history,'r')
		test_error=plt.plot(np.arange(self.epoch), J_history_test,'b')
		#plt.legend([train_error, test_error], ['Training Error', 'Testing Error'])
		plt.xlabel('Iterations')
		plt.ylabel('Cost Function')
		plt.show()
			

if __name__ == "__main__":

	#Intialise a single neuron neural network.
	neural_network = NeuralNetwork()

	print "Random starting synaptic weights: "
	print neural_network.synaptic_weights_O
	neural_network.loadDataAndInitProcessing()

	print "New synaptic weights after training: "
	#print neural_network.synaptic_weights_O
	#testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)
	#feature_tuning=FeatureTuning()
	#testing_set_inputs=feature_tuning.Transform3dTo1D(testX)
	# Test the neural network with a new situation.
	