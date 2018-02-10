# Q4
# c1 --> pos
# c2 --> neg
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
	DS2_c1_m1 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_c1_m1.csv', header = None).dropna(axis=1, how='any')
	DS2_c1_m1 = DS2_c1_m1.as_matrix().flatten()
	DS2_c1_m2 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_c1_m2.csv', header = None).dropna(axis=1, how='any')
	DS2_c1_m2 = DS2_c1_m2.as_matrix().flatten()
	DS2_c1_m3 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_c1_m3.csv', header = None).dropna(axis=1, how='any')
	DS2_c1_m3 = DS2_c1_m3.as_matrix().flatten()

	DS2_c2_m1 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_c2_m1.csv', header = None).dropna(axis=1, how='any')
	DS2_c2_m1 = DS2_c2_m1.as_matrix().flatten()
	DS2_c2_m2 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_c2_m2.csv', header = None).dropna(axis=1, how='any')
	DS2_c2_m2 = DS2_c2_m2.as_matrix().flatten()
	DS2_c2_m3 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_c2_m3.csv', header = None).dropna(axis=1, how='any')
	DS2_c2_m3 = DS2_c2_m3.as_matrix().flatten()

	DS2_Cov1 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_Cov1.csv', header = None).dropna(axis=1, how='any')
	DS2_Cov1 = DS2_Cov1.as_matrix()
	DS2_Cov2 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_Cov2.csv', header = None).dropna(axis=1, how='any')
	DS2_Cov2 = DS2_Cov2.as_matrix()
	DS2_Cov3 = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/hwk2_datasets_corrected/DS2_Cov3.csv', header = None).dropna(axis=1, how='any')
	DS2_Cov3 = DS2_Cov3.as_matrix()

	c1_m1 = pd.DataFrame(data = create_class(DS2_c1_m1, DS2_Cov1,2000)).sample(frac = 0.1)
	c1_m2 = pd.DataFrame(data = create_class(DS2_c1_m2, DS2_Cov2,2000)).sample(frac = 0.42)
	c1_m3 = pd.DataFrame(data = create_class(DS2_c1_m3, DS2_Cov3,2000)).sample(frac = 0.48)

	c1 = np.concatenate((np.concatenate((np.array(c1_m1),np.array(c1_m2))),np.array(c1_m3)))
	np.random.shuffle(c1)

	c2_m1 = pd.DataFrame(data = create_class(DS2_c2_m1, DS2_Cov1,2000)).sample(frac = 0.1)
	c2_m2 = pd.DataFrame(data = create_class(DS2_c2_m2, DS2_Cov2,2000)).sample(frac = 0.42)
	c2_m3 = pd.DataFrame(data = create_class(DS2_c2_m3, DS2_Cov3,2000)).sample(frac = 0.48)

	c2 = np.concatenate((np.concatenate((np.array(c2_m1),np.array(c2_m2))),np.array(c2_m3)))
	np.random.shuffle(c2)

	DS2 = pd.DataFrame(data = np.concatenate((c1,c2)))
	training_set = DS2.sample(frac = 0.7)
	test_set = DS2.drop(training_set.index)

	DS2.to_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS2.csv',index = False, header = False)
	training_set.to_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS2_train.csv',index = False, header = False)
	test_set.to_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS2_test.csv',index = False, header = False)


