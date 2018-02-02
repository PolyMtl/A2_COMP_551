 # Q1
import numpy as np 
import pandas as pd

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
	# generate samples of classes
	class_negative = create_class(DS1_m_0,DS1_cov,2000)
	class_positive = create_class(DS1_m_1,DS1_cov,2000)
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
	x = np.array(training_class_negative)
	y = np.array(training_class_positive)
	train_set = np.concatenate((x,y))
	# random shuffle
	np.random.shuffle(train_set)
	x = np.array(testing_class_negative)
	y = np.array(testing_class_positive)
	test_set = np.concatenate((x,y))
	# randome shuffle
	np.random.shuffle(test_set)
	
	# dont know what to do frm here onwards
	# get the last element of each row
	a = train_set[:,-1]
	c0 = train_set[a==1][:100]
	print c0