# Q3
import pandas as pd
import numpy as np
import math
import time
import heapq

def run_KNN(test_set,training_set,test_labels,training_labels, k):
	pred_label_arr = []
	for true_lable,test_vector in enumerate(test_set):
		distance_vec_matrix = []
		distance_vec_matrix = np.subtract(test_vector,training_set)
		
		distance_arr = []
		for index,dist_vec in enumerate(distance_vec_matrix):
			heapq.heappush(distance_arr, (np.linalg.norm(dist_vec), index))
		
		NN_labels = []
		for i in heapq.nsmallest(k,distance_arr):
			NN_labels.append(training_labels[i[1]])

		for label in NN_labels:
			count_0 = 0
			count_1 = 0
			if (label == 0):
				count_0 +=1
			if (label == 1):
				count_1 +=1

			if (count_1 >= count_0):
				pred_label = 1
			else:
				pred_label = 0
		
		pred_label_arr.append(pred_label)	
	
	true_pos, false_pos, false_neg, f_measure = np.zeros(4)
	
	for i in range(len(pred_label_arr)):
		if(pred_label_arr[i] == 1 and test_labels[i] == 1):
			true_pos +=1
		if(pred_label_arr[i] == 1 and test_labels[i] == 0):
			false_pos +=1
		if(pred_label_arr[i] == 0 and test_labels[i] == 1):
			false_neg +=1

	precision = true_pos/(true_pos+false_pos)
	recall = true_pos/(true_pos+false_neg)
	f_measure = 2*(precision*recall)/(precision+recall)

	return precision, recall, f_measure

if __name__ == "__main__":
	start = time.clock()
	# get training and test set
	training_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_train.csv', header = None).dropna(axis=1, how='any')
	test_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_test.csv', header = None).dropna(axis=1, how='any')
	
	# get labels for each set
	training_labels = np.array(training_set[[20]])
	test_labels = np.array(test_set[[20]])
	
	# only dealing with samples, without lables
	training_set = training_set.drop([20], axis = 1)
	test_set = test_set.drop([20], axis = 1)
	
	# convert to array for ease of use
	training_set = np.array(training_set)
	test_set = np.array(test_set)

	k = np.arange(1,21)
	for k in k:
		precision, recall, f_measure = run_KNN(test_set, training_set, test_labels, training_labels,k)
		print "k:	",k
		print "precision:	",precision
		print "recall:	",recall
		print "f_measure:	",f_measure		

	# print time.clock() - start



