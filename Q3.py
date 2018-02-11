# Q3 and Q5
import pandas as pd
import numpy as np
import math
import time
import heapq
import matplotlib.pyplot as plt

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
	
	true_pos, false_pos, false_neg, true_neg, f_measure = np.zeros(5)
	
	for i in range(len(pred_label_arr)):
		if(pred_label_arr[i] == 1 and test_labels[i] == 1):
			true_pos +=1
		if(pred_label_arr[i] == 1 and test_labels[i] == 0):
			false_pos +=1
		if(pred_label_arr[i] == 0 and test_labels[i] == 1):
			false_neg +=1
		if (pred_label_arr[i] == 0 and test_labels[i] == 0):
			true_neg +=1

	accuracy = (true_pos+true_neg)/(len(pred_label_arr))
	precision = true_pos/(true_pos+false_pos)
	recall = true_pos/(true_pos+false_neg)
	f_measure = 2*(precision*recall)/(precision+recall)

	return accuracy,precision, recall, f_measure

if __name__ == "__main__":
	# DS1
	# get training and test set
	DS1_training_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_train.csv', header = None).dropna(axis=1, how='any')
	DS1_test_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS1_test.csv', header = None).dropna(axis=1, how='any')
	
	# get labels for each set
	DS1_training_labels = np.array(DS1_training_set[[20]])
	DS1_test_labels = np.array(DS1_test_set[[20]])
	
	# only dealing with samples, without lables
	DS1_training_set = DS1_training_set.drop([20], axis = 1)
	DS1_test_set = DS1_test_set.drop([20], axis = 1)
	
	# convert to array for ease of use
	DS1_training_set = np.array(DS1_training_set)
	DS1_test_set = np.array(DS1_test_set)

	# # DS2
	# # get training and test set
	# DS2_training_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS2_train.csv', header = None).dropna(axis=1, how='any')
	# DS2_test_set = pd.read_csv(r'/Users/vivek/git/A2_COMP_551/Datasets/DS2_test.csv', header = None).dropna(axis=1, how='any')

	# # get labels for each set
	# DS2_training_labels = np.array(DS2_training_set[[20]])
	# DS2_test_labels = np.array(DS2_test_set[[20]])

	# # only dealing with samples, without lables
	# DS2_training_set = DS2_training_set.drop([20], axis = 1)
	# DS2_test_set = DS2_test_set.drop([20], axis = 1)

	# # convert to array for ease of use
	# DS2_training_set = np.array(DS2_training_set)
	# DS2_test_set = np.array(DS2_test_set)


	k_arr = np.arange(1,21)
	DS1_accuracy_arr = []
	DS1_precision_arr = []
	DS1_recall_arr = []
	DS1_f_measure_arr = []
	
	# DS2_accuracy_arr = []
	# DS2_precision_arr = []
	# DS2_recall_arr = []
	# DS2_f_measure_arr = []

	for k in k_arr:
		accuracy, precision, recall, f_measure = run_KNN(DS1_test_set, DS1_training_set, DS1_test_labels, DS1_training_labels,k)
		DS1_accuracy_arr.append(accuracy)
		DS1_precision_arr.append(precision)
		DS1_recall_arr.append(recall)
		DS1_f_measure_arr.append(f_measure)
		# accuracy, precision, recall, f_measure = run_KNN(DS2_test_set, DS2_training_set, DS2_test_labels, DS2_training_labels,k)
		# DS2_accuracy_arr.append(accuracy)
		# DS2_precision_arr.append(precision)
		# DS2_recall_arr.append(recall)
		# DS2_f_measure_arr.append(f_measure)


	DS1_accuracy_arr = np.array(DS1_accuracy_arr)
	DS1_best_accuracy = np.amax(DS1_accuracy_arr)
	DS1_best_accuracy_KVal = np.argmax(DS1_accuracy_arr)
	print "DS1_best_accuracy:	", DS1_best_accuracy, "Kval:	", DS1_best_accuracy_KVal
		
	DS1_precision_arr = np.array(DS1_precision_arr)
	DS1_best_precision = np.amax(DS1_precision_arr)
	DS1_best_precision_KVal = np.argmax(DS1_precision_arr)
	print "DS1_best_precision:	",DS1_best_precision,"	Kval:	", DS1_best_precision_KVal
	
	DS1_recall_arr = np.array(DS1_recall_arr)
	DS1_best_recall = np.amax(DS1_recall_arr)
	DS1_best_recall_KVal = np.argmax(DS1_recall_arr)
	print "DS1_best_recall:	",DS1_best_recall, "	Kval:	", DS1_best_recall_KVal
	
	DS1_f_measure_arr = np.array(DS1_f_measure_arr)
	DS1_best_f_measure = np.amax(DS1_f_measure_arr)
	DS1_best_f_measure_KVal = np.argmax(DS1_f_measure_arr)
	print "DS1_best_f_measure:	",DS1_best_f_measure, "	Kval:	",DS1_best_f_measure_KVal
	
	# DS2_accuracy_arr = np.array(DS2_accuracy_arr)
	# DS2_best_accuracy = np.amax(DS2_accuracy_arr)
	# DS2_best_accuracy_KVal = np.argmax(DS2_accuracy_arr)
	# print "DS2_best_accuracy:	", DS2_best_accuracy, "Kval:	", DS2_best_accuracy_KVal

	# DS2_best_precision = np.array(DS2_precision_arr)
	# DS2_best_precision = np.amax(DS2_precision_arr)
	# DS2_best_precision_KVal = np.argmax(DS2_precision_arr)
	# print "DS2_best_precision:	",DS2_best_precision,"	Kval:	", DS2_best_precision_KVal

	# DS2_best_recall = np.array(DS2_recall_arr)
	# DS2_best_recall = np.amax(DS2_recall_arr)
	# DS2_best_recall_KVal = np.argmax(DS2_recall_arr)
	# print "DS2_best_recall:	",DS2_best_recall, "	Kval:	", DS2_best_recall_KVal
	
	# DS2_best_f_measure = np.array(DS2_f_measure_arr)
	# DS2_best_f_measure = np.amax(DS2_f_measure_arr)
	# DS2_best_f_measure_KVal = np.argmax(DS2_f_measure_arr)
	# print "DS2_best_f_measure:	",DS2_best_f_measure, "	Kval:	",DS2_best_f_measure_KVal


	plt.figure(figsize=(20,8), dpi=80)
	plt.plot(k_arr, DS1_accuracy_arr, '-', label = 'Accuracy')
	plt.plot(k_arr, DS1_f_measure_arr, '-', label = 'F-Measure')
	plt.plot(k_arr, DS1_recall_arr, '-', label = 'Recall')
	plt.plot(k_arr, DS1_precision_arr, '-', label = 'Precision')
	plt.xlabel('K-Value')
	plt.ylabel('Percentage')
	plt.legend(loc=1)
	plt.title('KNN Evaluation Metrics - DS1')
	plt.show()



