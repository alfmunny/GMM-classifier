__author__ = 'Yuanchen'

""" Script to show some examples to use the gmm_train """
""" Please only run the script under current folder """

import sys
import os
sys.path.append('../../../../lernstift.new/')
import gmm_train as gtrain
import numpy as np
import operator
import json
import matplotlib.pyplot as plt
import pickle
from pen.classifier.learning import feature_processing
from sklearn import metrics
from sklearn import preprocessing

""" =================================== """
""" ==== Initialization ==== """
f_config = open('../tests/config_multiwriter_test.json', 'rb')
#f_data = open('Adham_Carina_David_Mario_Yesser_Boris_Jessica_Ancer_Deepak_Wahib_IMU_processed.pkl', 'rb')
#f_data = open('Adham_Carina_David_Mario_Yesser_Boris_Jessica_Ancer_Deepak_Wahib_pn_IMU_processed.pkl', 'rb')
f_data = open('multi_writer_model_on_Oct2_version1_on_pen_processed_feature.pkl', 'rb')
#f_data = open('./feature_resample.pkl', 'rb')
config = json.load(f_config)
data = pickle.load(f_data)

test_writer_list = config['writers']
test_word_list = config['selected_words']
test_letter_list = config['selected_classes']

""" =================================== """
""" ==== Example 0: Feature extraction ====="""
# raw_test, buffer_test = feature_processing.load_data_writer_word(test_writer_list, test_word_list)
# total_data = gtrain.feature_extraction(raw_test)
# f_feature = open('feature_resample.pkl', 'wb')
# pickle.dump(total_data, f_feature)

""" =================================== """
""" ==== Example 0: Generate final model ====="""
#writer_name = 'No Body'
#gtrain.gaussian_distribution(data, config, writer_name)


""" =================================== """
""" ==== Example 1: Calculate average accuracy for Multiwriter ====="""
""" output is a dict to save the result summary for all letters and all writers"""
""" like: {'a': array([65, 100]), 'b': array([75, 100])} """
#
# new_data = []
# for x in data:
#     new_data.append(x)
#     new_data[-1][1] = x[1][:, 0]
#
# new_data = preprocessing.normalize(data)

sorted_accuracy, over_all_test_label, over_all_predict_label = gtrain.calc_multiwriter_accuray(data, config, test_writer_list, test_letter_list)
accuracy_list = np.array([x[1] for x in sorted_accuracy])

print "=============Result Summary===================="
print "Selected classes: " + str(test_letter_list)
print "Selected writers: " + str(test_writer_list)
print "Avarage results of all letters: ", accuracy_list.mean()
print ("Confusion matrix:\n\n%s" %
       metrics.confusion_matrix(over_all_test_label, over_all_predict_label, test_letter_list))

""" ==================================== """
""" ==== Example 2: Calculate a average learning curve of all writers ==== """
# test_accuracy_list_sum = []
# train_accuracy_list_sum = []
# counter = 0
#
# for writer in test_writer_list:
#     train_dict, test_dict = gtrain.extract_data_multiwriter(data, config, writer)
#     """ please set start and step by real test, normally 0.02 and 0.02 """
#     ratio_list, test_accuracy_list, train_accuracy_list = gtrain.learning_curve(train_dict, test_dict, config)
#     #ratio_list, test_accuracy_list, train_accuracy_list = gtrain.learning_curve(train_dict, test_dict, 0.2, 0.2)
#     if counter == 0:
#         test_accuracy_list_sum = np.array(test_accuracy_list)
#         train_accuracy_list_sum = np.array(train_accuracy_list)
#     else:
#         test_accuracy_list_sum = test_accuracy_list_sum + np.array(test_accuracy_list)
#         train_accuracy_list_sum = train_accuracy_list_sum + np.array(train_accuracy_list)
#     counter += 1
#
# test_accuracy_list_sum = test_accuracy_list_sum/float(counter)
# train_accuracy_list_sum = train_accuracy_list_sum/float(counter)
#
# """==== Save the curve  ===="""
# curve_dict = {}
# curve_dict['ratio_list'] = ratio_list
# curve_dict['test_accuracy_list'] = test_accuracy_list_sum
# curve_dict['train_accuracy_list'] = train_accuracy_list_sum
#
# f_curve = open('./curve_data', 'wb')
# pickle.dump(curve_dict, f_curve)
# f_curve.close()
#
# """==== Plot the curve ===="""
# x1,x2,y1,y2 = plt.axis()
# plt.axis((x1,x2,0,1))
# plt.plot(ratio_list, test_accuracy_list_sum)
# plt.plot(ratio_list, train_accuracy_list_sum)
# plt.show()

""" ======================================= """
""" === Example 3: Calculate learning === """
#
# train_dict, test_dict = gtrain.extract_data_ratio(data, config, 0.6)
# ratio_list, test_accuracy_list, train_accuracy_list = gtrain.learning_curve(train_dict, test_dict, config)
#
# """==== Plot the curve ===="""
# x1,x2,y1,y2 = plt.axis()
# plt.axis((x1,x2,0,1))
# plt.plot(ratio_list, test_accuracy_list)
# plt.plot(ratio_list, train_accuracy_list)
# plt.show()
