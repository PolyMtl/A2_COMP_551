# Q2
# esitmate parameters of probabilistic LDA by maximizing likelihood function
# parameters of likelihood function: DS1_m_0, DS1_m_1, DS1_Cov, P(C_0), P(C_1)
# let P(C_0) = pi, P(C_1) = 1-pi 
import pandas as pd
import numpy as np

def get_class_samples(df):
	samples_C0 = []
	samples_C1 = []

	for i,vector in df.iterrows():
		vector = np.array(vector)
		if(vector[20] == 1):
			samples_C1.append(vector)
		if(vector[20] == 0):
			samples_C0.append(vector)

	samples_C0 = np.array(samples_C0)
	samples_C1 = np.array(samples_C1)
	return samples_C0, samples_C1

def calc_opt_pi(df):
	N1 = 0
	N2 = 0
	for i in (df[20]):
		if(i == 0):
			N1 = N1 + 1
		if(i == 1):
			N2 = N2 + 1

	return float(N1)/(N1+N2)

def calc_opt_mean_vectors(samples_C0, samples_C1):
	sum_C0 = np.zeros(20)
	sum_C1 = np.zeros(20)
	for j in samples_C0:
		j = j[:-1]
		sum_C0 = np.array(np.add(sum_C0,j))
	opt_mean_C0 = np.dot((float(1)/1400), sum_C0)

	for k in samples_C1:
		k = k[:-1]
		sum_C1 = np.array(np.add(sum_C1,k))
	opt_mean_C1 = np.dot((float(1)/1400), sum_C1)

	return opt_mean_C0, opt_mean_C1

def calc_opt_cov_matrix(samples_C0, samples_C1, opt_mean_C0, opt_mean_C1):
	neg_mean_C0 = np.dot(-1,opt_mean_C0)
	cov_C0 = np.zeros((20,20))
	opt_cov = np.zeros((20,20))
	for j in samples_C0:
		j = j[:-1]
		temp = np.add(j,neg_mean_C0)
		cov_C0 = np.add(np.outer(temp,temp),cov_C0)
	cov_C0 = np.dot((float(1)/1400),cov_C0)

	neg_mean_C1 = np.dot(-1,opt_mean_C1)
	cov_C1 = np.zeros((20,20))
	for j in samples_C0:
		j = j[:-1]
		temp = np.add(j,neg_mean_C1)
		cov_C0 = np.add(np.outer(temp,temp),cov_C1)
	cov_C1 = np.dot((float(1)/1400),cov_C1)

	opt_cov = np.add(cov_C0,cov_C1)
	return opt_cov

if __name__ == "__main__":
	# get training set
	training_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_train.csv', header = None).dropna(axis=1, how='any')
	# need to calc optimal pi = N1/(N1+N2)
	opt_pi = calc_opt_pi(training_set)
	samples_C0,samples_C1 = get_class_samples(training_set)
	opt_mean_C0, opt_mean_C1 = calc_opt_mean_vectors(samples_C0,samples_C1)
	opt_cov = calc_opt_cov_matrix(samples_C0,samples_C1,opt_mean_C0,opt_mean_C1)

	
	




