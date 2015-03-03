__author__ = 'Yuanchen'

""" A script for train and test dataset using Gaussian Mixture Density"""

import sys
import os
sys.path.append('../../../../lernstift.new/')
from pen.classifier.learning import feature_processing
from scipy.signal import resample
import pickle
import numpy as np
from sklearn import mixture
import json
import operator
from pen.classifier.learning import feature_processing

def gaussian_distribution(data, config, writer_name, ratio=None):
    """
    Train and export model and data for a particular writer.
    When generating the final model without indicating any writer, just give writer_name a string like "Nobody"
    :param data: feature set list
    :param config: configuration dictionary
    :param writer_name: writer name in string
    :param ratio: ratio to be used in extracting the data as train data
    :return: model, train dataset, testdataset
    """

    """ Initialization """
    data_all = data
    letter_list = config['selected_classes']

    if ratio is None:
        ratio = 1

    train_array = []
    test_array = []
    train_dict = {}
    test_dict = {}
    model_dict = {}
    data_dict = {}

    """ Extract data from train and test """
    print "================================="
    print "Tranning model by using " + writer_name + "'s data as test data"
    for l in letter_list:
        train_temp_array = []
        test_temp_array = []
        random_array = []
        for d in data_all:
            if d[0] == l and d[2] != writer_name and d[2] != 'Carina':
                train_temp_array.append(d[1])
            if d[0] == l and d[2] == writer_name:
                test_temp_array.append(d[1])

        random_index = np.random.permutation(range(0, int(len(train_temp_array)*ratio)))

        for index in random_index:
            random_array.append(train_temp_array[index])

        train_array = np.array(random_array)

        test_array = np.array(test_temp_array)
        temp_array = train_temp_array + test_temp_array
        data_array = np.array(temp_array)

        train_dict[l] = train_array
        test_dict[l] = test_array
        data_dict[l] = data_array

    f_data = open('./GMM_DATA.pkl', 'wb')
    f_test = open('./GMM_TEST.pkl', 'wb')
    f_train = open('./GMM_TRAIN.pkl', 'wb')

    pickle.dump(test_dict, f_test)
    pickle.dump(train_dict, f_train)
    pickle.dump(data_dict, f_data)

    """ Training with trainingdata """
    for l in letter_list:
        if l in train_dict and len(train_dict[l]) > 0:
        #  print train_dict[l]
            model = mixture.GMM(n_components=1, n_iter=100, covariance_type="full")
            model.fit(train_dict[l])
            model_dict[l] = model
        else:
            print "There are no training data for '" + l + "'"

    f_model = open('./GMM_MODEL.pkl', 'wb')
    pickle.dump(model_dict, f_model)

    f_data.close()
    f_train.close()
    f_test.close()
    f_model.close()
    print "================================="

    return model, train_dict, test_dict

def extract_data_multiwriter(data, config, writer_name, ratio=None):
    """
    Extract data for multiwriter tests.
    :param data: feature set list
    :param config: configuration dictionary
    :param writer_name: writer name in string
    :param ratio: ratio to be used in extracting the data as train data
    :return: train dataset, testdataset
    """

    """ Initialization """
    letter_list = config['selected_classes']
    data_all = data
    if ratio is None:
        ratio = 1

    train_dict = {}
    test_dict = {}
    model_dict = {}
    data_dict = {}

    """ Extract data from train and test """
    print "================================="
    print "Tranning model by using " + writer_name + "'s data as test data with ratio " + str(ratio)

    for l in letter_list:
        train_temp_array = []
        test_temp_array = []
        random_array = []
        for d in data_all:
            if d[0] == l and d[2] != writer_name and d[2] != 'Carina':
                train_temp_array.append(d[1])
            if d[0] == l and d[2] == writer_name:
                test_temp_array.append(d[1])

        #random_index = np.random.permutation(range(0, int(len(train_temp_array)*ratio)))
        np.random.shuffle(train_temp_array)
        # for index in random_index:
        #     random_array.append(train_temp_array[index])

        #train_array = np.array(random_array)
        train_array = train_temp_array[0:(len(train_temp_array)*ratio-1)]

        test_array = np.array(test_temp_array)
        temp_array = train_temp_array + test_temp_array
        data_array = np.array(temp_array)

        train_dict[l] = train_array
        test_dict[l] = test_array
        data_dict[l] = data_array

    """ Exporting the data """
    #f_data = open('./GMM_DATA.pkl', 'wb')
    #f_test = open('./GMM_TEST.pkl', 'wb')
    #f_train = open('./GMM_TRAIN.pkl', 'wb')

    #pickle.dump(test_dict, f_test)
    #pickle.dump(train_dict, f_train)
    #pickle.dump(data_dict, f_data)

    return train_dict, test_dict

def extract_data_ratio(data, config, ratio=None):
    """
    Extract data for multiwriter tests.
    :param data: feature set list
    :param config: configuration dictionary
    :param ratio: ratio to be used in extracting the data as train data
    :return: train dataset, testdataset
    """

    """ Initialization """
    letter_list = config['selected_classes']
    data_all = data
    if ratio is None:
        ratio = 1

    train_dict = {}
    test_dict = {}
    model_dict = {}
    data_dict = {}

    """ Extract data from train and test """
    print "================================="

    for l in letter_list:
        train_temp_array = []
        for d in data_all:
            if d[0] == l and d[2] != 'Carina':
                train_temp_array.append(d[1])

        np.random.shuffle(train_temp_array)

        train_array = train_temp_array[:(int(len(train_temp_array)*ratio)-1)]
        test_array = train_temp_array[int((len(train_temp_array)*ratio)-1):]

        train_dict[l] = train_array
        test_dict[l] = test_array

    train_number = 0
    test_number = 0
    for n, d in train_dict.items():
        train_number = train_number + len(d)
    print "Train data:" + str(train_number)

    for n, d in test_dict.items():
        test_number = test_number + len(d)
    print "Test data:" + str(test_number)

    #f_test = open('./GMM_TEST.pkl', 'wb')
    #f_train = open('./GMM_TRAIN.pkl', 'wb')

    #pickle.dump(test_dict, f_test)
    #pickle.dump(train_dict, f_train)

    return train_dict, test_dict

def train(train_dict, config):
    """
    Train the model using train dictionary
    :param train_dict: feature set list
    :param config: configuration dictionary
    :return: model dictionary
    """
    letter_list = config['selected_classes']
    n_components = config['gmm_options']['n_components']
    n_iter = config['gmm_options']['n_iter']
    covariance_type = config['gmm_options']['covariance_type']
    #covariance_type = 'diag'
    model_dict = {}

    """ Training with trainingdata """
    for l in letter_list:
        #model = mixture.GMM(n_components=1, n_iter=100, covariance_type="full")
        #model = mixture.GMM(n_components=5, n_iter=100, covariance_type="diag")
        if l in train_dict and len(train_dict[l]) > 0:
            model = mixture.GMM(n_components=n_components, n_iter=n_iter, covariance_type=covariance_type)
            model.fit(train_dict[l])
            model_dict[l] = model
        else:
            print "There are no training data for '" + l + "'"

    #f_model = open('./GMM_MODEL.pkl', 'wb')
    #pickle.dump(model_dict, f_model)

    return model_dict

def train_unsupervised(data, name, n_components, covariance_type):

    """
    Train the model using train dictionary
    :param data: list : feature set list
    :param name: string : the name for the output file
    :param n_components: integer : the number of components for gaussian model
    :param covariance_type: string : full or diag
    :return: model dictionary
    """

    output_file = name + '.pkl'
    #data = np.array(data)
    model = mixture.GMM(n_components=n_components, covariance_type=covariance_type)
    model.fit(data)

    f_model = open(output_file, 'wb')
    pickle.dump(model, f_model)

    return model

def calc_accuracy(model_dict, test_dict):
    """
    Calculate the accuracy using test dictionary and model dictionary
    :param model_dict: model dictionary
    :param test_dict: test dictionary
    :return:
      all_prob: list for each accuracy of every letter
      reuslt_dict: dict for each letter and its results, which are correct numbers and whole numbers.
    """

    """ Calculate the result """

    all_prob = []
    result_dict = {}
    test_label = []
    predict_label = []

    for t_name, t in test_dict.items():
        result = []
        index = []
        hype_dict = {}
        sum = len(t)
        counter = 0
        letter = t_name
        for p in t:
            test_label.append(t_name)
            high_score = -100000
            for m_name, m in model_dict.items():
                score = m.score([p])
                if score > high_score:
                    high_score = score
                    hypo = m_name
            result.append(hypo)
            predict_label.append(hypo)
            if hypo == letter:
                counter += 1
        all_letters = list(set(result))
        for l in all_letters:
            hype_dict[l] = result.count(l)

        sorted_hype_dict = sorted(hype_dict.iteritems(), key=operator.itemgetter(1))
        sorted_hype_dict.reverse()

        if sum != 0:
          prob = float(counter)/sum
          print str(letter) + "("+ str(counter) + "/" + str(sum) + ")" + " ==> Accuracy: " + str(prob),
          print sorted_hype_dict
          all_prob.append(prob)
          result_dict[letter] = np.array([counter, sum])

    """ Print the average accuracy"""

    all_prob = np.array(all_prob)
    print "Average accuracy is: " + str(all_prob.mean())
    print "================================="

    return all_prob, result_dict, test_label, predict_label

def calc_multiwriter_accuray(data, config, writer_list, letter_list):
    '''
    data: all the feature data
    config: configuration
    writer_list: a list contains all the writers, which to be tested
    letter_list: a list contains all the letters, which to be tested
    '''

    output = {}

    """ accuracy is a dict to save the accuracy for all letters """
    accuracy = {}
    over_all_test_label = []
    over_all_predict_label = []

    """ Initialize the output """
    for l in letter_list:
        output[l]=np.array([0,0])

    """ Run the test """
    for writer in writer_list:

        """ Train model for selected writer"""
        train_dict, test_dict = extract_data_multiwriter(data, config, writer)
        model_dict = train(train_dict, config)

        """ Test using selected writer's data """
        all_accuracy, result, test_label, predict_label = calc_accuracy(model_dict, test_dict)
        over_all_test_label = over_all_test_label + test_label
        over_all_predict_label = over_all_predict_label + predict_label

        """ Sum the result with the last """
        for l in letter_list:
            if l in result:
                output[l] = np.add(output[l],result[l])

    """ Calculate the avarage accuracy for all letters """
    for l, n in output.items():
        if n[1] != 0:
            accuracy[l] = float(n[0])/n[1]

    """ Sort the dict and save to an array """
    sorted_accuracy = sorted(accuracy.iteritems(), key=operator.itemgetter(1))

    """ ==== Print the result ==== """
    for item in sorted_accuracy:
        print 'Symbol:', str(item[0]), ', accuracy:', str(item[1])

    return sorted_accuracy, over_all_test_label, over_all_predict_label

def learning_curve(train_dict, test_dict, config, start=0.02, step=0.02):
    """
    Calculate the learning curve
    :param train_dict: train dictionary
    :param test_dict: test dictionary
    :param config: configuration dictionary
    :optional param start: the starting point of the curve in percentage, which is a ratio to set the used train data
    :optional param step: the step of curve for each iteration
    :return:
      all_prob: list for each accuracy of every letter
      reuslt_dict: dict for each letter and its results, which are correct numbers and whole numbers.
    """
    ratio_list = np.arange(start, 1 + step, step)
    train_dict_part = {}
    test_accuracy_list = []
    train_part_accuracy_list = []

    for ratio in ratio_list:
        for symbol, t_data in train_dict.items():
            train_dict_part[symbol] = t_data[0: int(len(t_data)*ratio)]

        #train_dict, test_dict = extract_train_data(writer_name, ratio)
        model_dict = train(train_dict_part, config)
        print "================================="
        print "Test results using " + str(ratio*100) + "% of the training data"

        """ Calculate the result for test data """
        all_prob_test, result_dict_test, test_label_test, predict_label_test = calc_accuracy(model_dict, test_dict)
        """ Print the average accuracy for test data """
        all_prob_test = np.array(all_prob_test)
        accuracy_test = round(all_prob_test.mean(), 4)
        print "Average accuracy of test data is: " + str(accuracy_test)


        """ Calculate the result for train data """
        all_prob_train_part, result_dict_train_part, test_label_train, predict_label_train = calc_accuracy(model_dict, train_dict_part)
        """ Print the average accuracy for train data """
        all_prob_train_part = np.array(all_prob_train_part)
        accuracy_train_part = round(all_prob_train_part.mean(), 4)
        print "Average accuracy of train data is: " + str(accuracy_train_part)
        print "================================="

        test_accuracy_list.append(accuracy_test)
        train_part_accuracy_list.append(accuracy_train_part)

    return ratio_list, test_accuracy_list, train_part_accuracy_list

def feature_extraction(raw_data):
    data = raw_data
    for d in data:
        tmp = resample(d[1][:, 0:6], 65)
        d[1] = tmp
    return data

def data_balance(data):
    balanced_data = data
    return balanced_data

def main(data, config, writer_name):
    gaussian_distribution(data, config, writer_name)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
