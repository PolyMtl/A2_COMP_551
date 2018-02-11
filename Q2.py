# Q2 & Q5
# need to model P(C_k | x)
# model P(x|C_k) and P(C_k) and use Bayes law
# in probabilistic LDA:
# 	class conditional probabilities are guassian
# 	all classes share same cov matrix
# esitmate parameters of probabilistic LDA by maximizing likelihood function
# parameters of likelihood function: DS1_m_0, DS1_m_1, DS1_Cov, P(C_0), P(C_1)
# let P(C_0) = pi, P(C_1) = 1-pi 

# C0 ---> label = 0, C1--->label =1
import pandas as pd
import numpy as np
import math

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
	opt_mean_C0 = np.divide(sum_C0, len(samples_C0))

	for k in samples_C1:
		k = k[:-1]
		sum_C1 = np.array(np.add(sum_C1,k))
	opt_mean_C1 = np.divide(sum_C1, len(samples_C1))

	return opt_mean_C0, opt_mean_C1

def calc_opt_cov_matrix(samples_C0, samples_C1, opt_mean_C0, opt_mean_C1):
	cov_C0 = np.zeros((20,20))
	for j in samples_C0:
		j = j[:-1]
		tempj = np.zeros(20)
		tempj = np.subtract(j,opt_mean_C0)
		cov_C0 = cov_C0 + np.outer(tempj,tempj) 

	cov_C1 = np.zeros((20,20))
	for k in samples_C1:
		k = k[:-1]
		tempk = np.zeros(20)
		tempk = np.subtract(k,opt_mean_C1)
		cov_C1 = cov_C1 + np.outer(tempk,tempk) 
	
	opt_cov = np.zeros((20,20))
	opt_cov = np.divide(np.add(cov_C0,cov_C1),2800)
	return opt_cov

def run_prob_LDA(opt_pi,opt_cov,opt_mean_C0,opt_mean_C1):
	w = np.dot(np.linalg.inv(opt_cov),np.subtract(opt_mean_C1,opt_mean_C0))
	w_0 = np.divide((np.dot(np.dot(opt_mean_C1.T,np.linalg.inv(opt_cov)),opt_mean_C1)),-2) + np.divide((np.dot(np.dot(opt_mean_C0.T,np.linalg.inv(opt_cov)),opt_mean_C0)),2) + np.log(opt_pi/(1-opt_pi))

	return (w,w_0)

def calc_best_fit_measures(test_set, w, w_0):
	true_pos, false_pos, false_neg, f_measure = np.zeros(4)
	
	for i,test_sample in test_set.iterrows():
		test_sample = np.array(test_sample)
		true = test_sample[20]
		test_sample = test_sample[:-1]
		prediction = np.dot(DS1_w,test_sample) + DS1_w_0
		if(prediction>=(0.5)):
			if(true == 1):
				true_pos +=1
			elif(true == 0):
				false_pos +=1
		elif(prediction<(0.5) and true==1):
			false_neg +=1

	precision = true_pos/(true_pos+false_pos)
	recall = true_pos/(true_pos+false_neg)
	f_measure = 2*(precision*recall)/(precision+recall)

	return (precision, recall, f_measure)


if __name__ == "__main__":
	# DS1
	# get training set
	DS1_traning_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_train.csv', header = None).dropna(axis=1, how='any')

	DS1_opt_pi = calc_opt_pi(DS1_traning_set)
	DS1_samples_C0,DS1_samples_C1 = get_class_samples(DS1_traning_set)
	DS1_opt_mean_C0, DS1_opt_mean_C1 = calc_opt_mean_vectors(DS1_samples_C0,DS1_samples_C1)
	DS1_opt_cov = calc_opt_cov_matrix(DS1_samples_C0,DS1_samples_C1,DS1_opt_mean_C0,DS1_opt_mean_C1)
	DS1_w,DS1_w_0 = run_prob_LDA(DS1_opt_pi,DS1_opt_cov,DS1_opt_mean_C0,DS1_opt_mean_C1)

	# get test set
	DS1_test_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_test.csv',header = None).dropna(axis=1, how='any')
	
	precision_DS1, recall_DS1, f_measure_DS1 = calc_best_fit_measures(DS1_test_set, DS1_w, DS1_w_0)
	print "DS1:	"
	print "precision:	", precision_DS1
	print "recall:	", recall_DS1
	print "f_measure:	", f_measure_DS1

	# DS2
	# get training set
	DS2_training_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS2_train.csv', header = None).dropna(axis=1, how='any')
	
	DS2_opt_pi = calc_opt_pi(DS2_training_set)
	DS2_samples_C0, DS2_samples_C1 = get_class_samples(DS2_training_set)
	DS2_opt_mean_C0, DS2_opt_mean_C1 = calc_opt_mean_vectors(DS2_samples_C0, DS2_samples_C1)
	DS2_opt_cov = calc_opt_cov_matrix(DS2_samples_C0,DS2_samples_C1,DS2_opt_mean_C0,DS2_opt_mean_C1)
	DS2_w,DS2_w_0 = run_prob_LDA(DS2_opt_pi,DS2_opt_cov,DS2_opt_mean_C0,DS2_opt_mean_C1)

	# get test set
	DS2_test_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS2_test.csv',header = None).dropna(axis=1, how='any')
	
	precision_DS2, recall_DS2, f_measure_DS2 = calc_best_fit_measures(DS2_test_set, DS2_w, DS2_w_0)
	print "DS2:	"
	print "precision:	", precision_DS2
	print "recall:	", recall_DS2
	print "f_measure:	", f_measure_DS2







