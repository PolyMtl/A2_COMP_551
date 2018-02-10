 # Q1
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def create_class(mean_vector, cov_matrix, num_samples):
	# returns a num_samples of (N X 1 vectors) where N is len(mean_vector) = len(cov_matrix)
	class_matrix = []
	for i in range(num_samples):
		class_matrix.append(np.random.multivariate_normal(mean_vector,cov_matrix))
	class_matrix = np.array(class_matrix)
	return class_matrix

if __name__ == "__main__":
	# get dataset --> conver to vectors and matrices
	DS1_m_0 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS1_m_0.csv', header = None).dropna(axis=1, how='any')
	DS1_m_0 = DS1_m_0.as_matrix().flatten()
	DS1_m_1 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS1_m_1.csv', header = None).dropna(axis=1, how='any')
	DS1_m_1 = DS1_m_1.as_matrix().flatten()
	DS1_cov = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS1_cov.csv', header = None).dropna(axis=1, how='any')
	DS1_cov = DS1_cov.as_matrix()
	
	# generate 2000 samples of each class
	class_negative = create_class(DS1_m_0,DS1_cov,2000)
	class_positive = create_class(DS1_m_1,DS1_cov,2000)

	# need to add respective labels as the 21st column?
	class_negative = np.append(class_negative, np.zeros((2000,1)), axis=1)
	class_positive = np.append(class_positive, np.ones((2000,1)), axis=1)

	# converted both classes to panads datafarme
	class_negative_df = pd.DataFrame(data = class_negative)
	class_positive_df = pd.DataFrame(data = class_positive)
	# create training and test set
	training_class_negative = class_negative_df.sample(frac = 0.7)
	testing_class_negative = class_negative_df.drop(training_class_negative.index)
	# create training and test set
	training_class_positive = class_positive_df.sample(frac = 0.7)
	testing_class_positive = class_positive_df.drop(training_class_positive.index)

	# create combined training set and test set (put both classes training sets and test sets together)
	train_set = np.concatenate((np.array(training_class_negative),np.array(training_class_positive)))
	test_set = np.concatenate((np.array(testing_class_negative),np.array(testing_class_positive)))
	# randome shuffle
	np.random.shuffle(train_set)
	np.random.shuffle(test_set)
	
	DS1 = np.concatenate((train_set,test_set))
	# write training and test set to csv
	pd.DataFrame(data = train_set).to_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_train.csv', index = False, header = False)
	pd.DataFrame(data = test_set).to_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_test.csv', index = False, header = False)
	pd.DataFrame(data = DS1).to_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1.csv',index = False, header = False)
	
	# ***** simple visualization of the two classes (from training set) ****** 
	# # get all labels of training set 
	# labels = train_set[:,-1]
	# # create new list of 100 samples where they are in class negative/positive
	# c_neg = train_set[labels == -1][:100]
	# c_pos = train_set[labels == 1][:100]
	
	# # plotting the 1st feat against the 2nd feat for both classes
	# plt.title('Simple visualization of classes') 
	# plt.scatter(c_neg[:,0], c_neg[:,1], c = 'b', label = 'class_negative')
	# plt.scatter(c_pos[:,0], c_pos[:,1], c = 'r', label = 'class_positive')
	# plt.legend(loc = 0)
	# plt.show()

