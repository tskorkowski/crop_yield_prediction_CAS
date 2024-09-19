# To execute: python3 train_NN.py -k keyfile.key

import argparse
import csv
from nnet_LSTM import *
from nnet_CNN import *
import sys
import os
import logging
from util import Progbar
from sklearn.metrics import r2_score
import getpass

#from GP import GaussianProcess
from scipy.stats.stats import pearsonr  
from constants import GBUCKET, DATASETS

#import gspread
#from spreadsheet_auth import get_credentials

t = time.localtime()
timeString  = time.strftime("%Y-%m-%d_%H-%M-%S", t)

AUTH_FILE_NAME = "static_data_files/crop-yield-2049f6b103d2.json"
input_data_dir = None
output_data_dir = None
worksheet = None

def read_key_file(key_file_path):
    params = {}
    with open(key_file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):  # Ignore comments and empty lines
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    params[key] = value.split("#")[0].strip()  # This also removes comments from the value, if any
    return params

def augment_data(inputs, labels, low_yield_threshold=1.8, noise_level=0.01, replication_factor=5):
    inputs = inputs.astype(np.float32)
    labels = labels.astype(np.float32)

    low_yield_indices = np.where(labels < low_yield_threshold)[0]
    augmented_inputs = np.copy(inputs)
    augmented_labels = np.copy(labels)

    for index in low_yield_indices:
        for _ in range(int(replication_factor)):  # replicate 'replication_factor' times
            noisy_input = inputs[index] + noise_level * np.random.randn(*inputs[index].shape)
            augmented_inputs = np.append(augmented_inputs, noisy_input[np.newaxis, ...], axis=0)
            augmented_labels = np.append(augmented_labels, labels[index])

    return augmented_inputs, augmented_labels

def file_generator(inputs_data_path, labels_data_path, batch_size, add_data_path=None, add_labels_path=None, permuted_band=-1):
    current_batch_labels = []
    current_batch_inputs = []
    inputs_data = np.load(inputs_data_path)['data']
    labels_data = np.load(labels_data_path)['data']

    if data_augmentation:
        inputs_data, labels_data = augment_data(inputs_data, labels_data, low_yield_threshold=yield_treshold, noise_level=noise_level, replication_factor=replication_factor)

    if permuted_band != -1:
        n_training_examples = inputs_data.shape[0]
        examples_permutation = np.random.permutation(n_training_examples)
        inputs_data[:, :, :, permuted_band] = inputs_data[examples_permutation, :, :, permuted_band]

    assert(len(inputs_data) == len(labels_data))
    if add_data_path is not None:
        inputs_data_2 = np.load(add_data_path)['data']
        inputs_data = np.append(inputs_data, inputs_data_2, axis = 0)
    if add_labels_path is not None:
        labels_data_2 = np.load(add_labels_path)['data']
        labels_data = np.append(labels_data, labels_data_2, axis = 0)
    
    while(len(inputs_data) < batch_size):
        #need to pad
        inputs_data = np.append(inputs_data, inputs_data, axis = 0)
        labels_data = np.append(labels_data, labels_data, axis = 0)
        print("appended to pad")
    for idx in range(len(inputs_data)):
       current_batch_inputs.append(inputs_data[idx])
       current_batch_labels.append(labels_data[idx]) 
       if len(current_batch_labels) == batch_size:
            yield current_batch_inputs, current_batch_labels
            current_batch_inputs = []
            current_batch_labels = []

def return_train_batch(inputs_data_path, labels_data_path, batch_size, permuted_band = -1):
    try:
        inputs_data = np.load(inputs_data_path)['data']
    except EOFError as e:
        print(f"Error loading {inputs_data_path}: {e}. File may be empty or corrupted.")
        return None  
    if os.path.exists(labels_data_path) and os.path.getsize(labels_data_path) > 0:
        try:
            labels_data = np.load(labels_data_path)['data']
        except EOFError as e:
            print(f"Error loading {labels_data_path}: {e}")

    if data_augmentation:
        inputs_data, labels_data = augment_data(inputs_data, labels_data, low_yield_threshold=yield_treshold, noise_level=noise_level, replication_factor=replication_factor)

    if permuted_band != -1:
        n_training_examples = inputs_data.shape[0]
        examples_permutation = np.random.permutation(n_training_examples)
        inputs_data[:, :, :, permuted_band] = inputs_data[examples_permutation, :, :, permuted_band]
    while True:
        indices = np.random.randint(0, len(inputs_data), size = batch_size)
        histograms = np.array([inputs_data[x] for x in indices])
        yields = np.array([labels_data[x] for x in indices])
        yield histograms, yields

def create_save_files(train_dir, train_name):
    scores_file = os.path.join(train_dir, 'scores_' + train_name + '.txt')
    sess_file = os.path.join(train_dir, 'weights', 'model.weights')
    train_predictions_file = os.path.join(train_dir, 'train_predictions.npz')
    dev_predictions_file = os.path.join(train_dir, 'dev_predictions.npz')
    test_predictions_file = os.path.join(train_dir, 'test_predictions.npz')
    model_w_file = os.path.join(train_dir, 'model_w.npz')
    model_b_file = os.path.join(train_dir, 'model_b.npz')
    train_feature_file = os.path.join(train_dir, 'train_features.npz')
    dev_feature_file = os.path.join(train_dir, 'dev_features.npz')
    test_feature_file = os.path.join(train_dir, 'test_features.npz')
    return scores_file, sess_file, train_predictions_file, test_predictions_file, dev_predictions_file, model_w_file, model_b_file, train_feature_file, test_feature_file, dev_feature_file


def end_and_output_results(worksheet, train_RMSE_min, train_R2_max, dev_RMSE_min, dev_R2_max, test_RMSE_min, test_R2_max):
    test_RMSE_to_append = ['best test_RMSE_min', test_RMSE_min]
    test_R2_to_append = ['best test_R2_max',test_R2_max]
    #worksheet_append_wrapper(worksheet, test_RMSE_to_append)
    #worksheet_append_wrapper(worksheet, test_R2_to_append)

    return test_RMSE_min, test_R2_max


def run_NN(model, sess, directory, CNN_or_LSTM, config, output_google_doc, restrict_iterations = None, rows_to_add_to_google_doc = None, permuted_band = -1):
    input_data_dir = directory
    train_name = "train_{}_{}_{}_{}_{}_{}".format(CNN_or_LSTM,config.lr, config.drop_out, config.train_step, timeString, sys.argv[1].replace('/','_'))
    train_dir = os.path.expanduser(os.path.join('Training', train_name))
    output_data_dir = train_dir
    os.mkdir(train_dir)
    train_logfile = os.path.join(train_dir, train_name + '.log')
    logging.basicConfig(filename=train_logfile,level=logging.DEBUG)
    model.writer = tf.compat.v1.summary.FileWriter(train_dir, graph=tf.compat.v1.get_default_graph())

    #google drive set-up
    dataset_name = directory[directory.find('/') + 1:]
    """
    worksheet_name = CNN_or_LSTM + "_" + dataset_name + "_" + timeString
    worksheet = authorize_gspread(output_google_doc, worksheet_name, True)
    if rows_to_add_to_google_doc is not None:
        for row in rows_to_add_to_google_doc:
            worksheet.append_row(row)
    if CNN_or_LSTM == "LSTM":
        worksheet.append_row(['learning rate', 'drop out', 'train step', 'loss_lambda', 'lstm_H', 'dense', 'dataset', 'NN', "NN_output_dir", "data_input_dir", "permuted_band"])
        worksheet.append_row([str(config.lr),str(config.drop_out),str(config.train_step), str(config.loss_lambda), str(config.lstm_H), str(config.dense),sys.argv[1].replace('/','_'), CNN_or_LSTM, train_dir.split('nnet_data/')[-1], sys.argv[1], str(permuted_band)])
    else:
        worksheet.append_row(['learning rate', 'drop out', 'train step', 'loss_lambda', 'dataset', 'NN', "NN_output_dir", "data_input_dir"])
        worksheet.append_row([str(config.lr),str(config.drop_out),str(config.train_step), str(config.loss_lambda), sys.argv[1].replace('/','_'), CNN_or_LSTM, train_dir.split('nnet_data/')[-1], sys.argv[1]])

    worksheet.append_row(['Dataset','ME', 'RMSE', 'R2', 'min RMSE', 'R2 for best RMSE', 'correlation_coeff', 'correlation_coeff var' ,'training loss'])
    """

    scores_file, sess_file, train_predictions_file, test_predictions_file, dev_predictions_file, model_w_file, model_b_file, train_feature_file, test_feature_file, dev_feature_file  = create_save_files(train_dir, train_name)

    # load data to memory
    train_data_file = os.path.join(input_data_dir, 'train_hists.npz') 
    train_labels_file = os.path.join(input_data_dir, 'train_yields.npz')
    dev_data_file = os.path.join(input_data_dir, 'dev_hists.npz') 
    dev_labels_file = os.path.join(input_data_dir, 'dev_yields.npz')
    test_data_file =  os.path.join(input_data_dir, 'test_hists.npz')
    test_labels_file = os.path.join(input_data_dir, 'test_yields.npz')

    dev_available = os.path.exists(dev_data_file)
    test_available = os.path.exists(test_data_file)
    
    summary_train_loss = []
    summary_eval_loss = []
    summary_RMSE = []
    summary_ME = []
    summary_R2 = []
    
    train_RMSE_min = 1e10
    test_RMSE_min = 1e10
    dev_RMSE_min = 1e10
    
    train_R2_max = 0
    test_R2_max = 0
    dev_R2_max = 0
    
    prev_train_loss = 1e10

    train_pred_file = open(os.path.join(train_dir, 'train_pred.csv'), 'w', newline='')
    train_real_file = open(os.path.join(train_dir, 'train_real.csv'), 'w', newline='')
    dev_pred_file = open(os.path.join(train_dir, 'dev_pred.csv'), 'w', newline='')
    dev_real_file = open(os.path.join(train_dir, 'dev_real.csv'), 'w', newline='')
    test_pred_file = open(os.path.join(train_dir, 'test_pred.csv'), 'w', newline='')
    test_real_file = open(os.path.join(train_dir, 'test_real.csv'), 'w', newline='')
    train_loss_file = open(os.path.join(train_dir, 'train_loss.csv'), 'w', newline='') 

    train_pred_writer = csv.writer(train_pred_file)
    train_real_writer = csv.writer(train_real_file)
    dev_pred_writer = csv.writer(dev_pred_file)
    dev_real_writer = csv.writer(dev_real_file)
    test_pred_writer = csv.writer(test_pred_file)
    test_real_writer = csv.writer(test_real_file)
    train_loss_writer = csv.writer(train_loss_file)

    try:
        count = 0
        target = 25
        prog = Progbar(target=target)
        if load_weights: config.lr/=100
        rmse_test_save = []
        
        #TRAINING PORTION
        for i in range(num_iters+1):
            # if i==3000 and not load_weights:
            #     config.lr/=10
            # if i==8000 and not load_weights:
            #     config.lr/=10
            batch = next(return_train_batch(train_data_file, train_labels_file, config.B, permuted_band = permuted_band))
            _, train_loss, summary, loss_summary = sess.run([model.train_op, model.loss, model.summary_op, model.loss_summary_op], feed_dict={
            # train_loss, summary, loss_summary = sess.run([model.loss, model.summary_op, model.loss_summary_op], feed_dict={                    # This prevents the weights from updating. 
                model.x: batch[0],
                model.y: batch[1],
                model.lr: config.lr,
                model.keep_prob: config.drop_out
                })
            prog.update(count + 1, [("train loss", train_loss)])
            count += 1
            if (i % 200):
                model.writer.add_summary(summary, i)
            else:
                model.writer.add_summary(loss_summary, i)

            train_loss_writer.writerow([train_loss])

            if i % target == 0:
                count = 0
                prog = Progbar(target=target)
                print("finished " + str(i))
                train_pred = []
                train_real = []
                train_features = []
                for batch in file_generator(train_data_file, train_labels_file, config.B, permuted_band = permuted_band):
                    pred_temp, feature, weight, bias = sess.run([model.pred, model.feature, model.dense_W, model.dense_B], feed_dict={
                        model.x: batch[0],
                        model.y: batch[1],
                        model.keep_prob: 1
                        })
                    train_pred.append(pred_temp)
                    train_features.append(feature)
                train_pred=np.concatenate(train_pred)
                train_features = np.concatenate(train_features)
                train_real = np.load(train_labels_file)['data'][:len(train_pred)]
                train_pred = train_pred[:len(train_real)]
                print("Train: train_real: ", train_real)
                print("Train: train_pred: ", train_pred)
                RMSE=np.sqrt(np.mean((train_pred-train_real)**2))
                ME=np.mean(train_pred-train_real)
                sklearn_r2 = r2_score(train_real, train_pred)

                correlation_coeff = pearsonr(train_pred, train_real)
                if i<10000: train_real_writer.writerow(train_real)
                train_pred_writer.writerow(train_pred)

                # if RMSE < train_RMSE_min:
                #     print("Found a new training RMSE minimum")
                #     train_RMSE_min = RMSE
                #     model.saver.save(sess, sess_file)  # Save the TensorFlow model
                #     # You might also want to save predictions, weights, features here, similar to what you do for the dev dataset
                #     # For example:
                #     np.savez(train_predictions_file, data=train_pred)
                #     print()
                #     np.savez(model_w_file, data=weight)
                #     np.savez(model_b_file, data=bias)
                #     np.savez(train_feature_file, data=train_features)

                if RMSE < train_RMSE_min:
                    train_RMSE_min = RMSE
                    train_R2_max = sklearn_r2
                """
                try:
                    worksheet = authorize_gspread(output_google_doc, worksheet_name)
                except:
                    time.sleep(3)
                    worksheet = authorize_gspread(output_google_doc, worksheet_name)
                """
                 
                print('Train set','test ME',ME, 'train RMSE',RMSE,'train R2',sklearn_r2,'train RMSE_min', train_RMSE_min,'train R2 for min RMSE',train_R2_max)
                logging.info('Train set train ME %f train RMSE %f train R2 %f train RMSE min %f train_R2_for_min_RMSE %f',ME, RMSE, sklearn_r2, train_RMSE_min,train_R2_max)
                
                line_to_append = ['Train',str(ME), str(RMSE), str(sklearn_r2), str(train_RMSE_min), str(train_R2_max), str(correlation_coeff[0]), str(correlation_coeff[1]), str(train_loss)]
                #worksheet_append_wrapper(worksheet, line_to_append)
                
                prev_train_loss = train_loss

                if dev_available:
                    #print scores on dev set
                    pred = []
                    real = []
                    dev_features = []
                    for batch in file_generator(dev_data_file, dev_labels_file, config.B, permuted_band = permuted_band):
                        pred_temp, feature = sess.run([model.pred, model.feature], feed_dict={
                            model.x: batch[0],
                            model.y: batch[1],
                            model.keep_prob: 1
                            })
                        pred.append(pred_temp)  
                        dev_features.append(feature)
                    pred=np.concatenate(pred)
                    dev_features = np.concatenate(dev_features)
                    real = np.load(dev_labels_file)['data']
                    pred = pred[:len(real)]
                    real = real[:len(pred)]
                    if i<10000: dev_real_writer.writerow(train_real)
                    dev_pred_writer.writerow(train_pred)
                    RMSE=np.sqrt(np.mean((pred-real)**2))
                    ME=np.mean(pred-real)
                    sklearn_r2 = r2_score(real, pred)
                    if len(pred)>2:
                        correlation_coeff = pearsonr(pred, real)
                    else:
                        correlation_coeff = [np.nan,np.nan]
                    
                    found_min = False
                    # if RMSE < dev_RMSE_min:
                    #     print("Found a new dev RMSE minimum")
                    #     found_min = True
                    #     dev_RMSE_min = RMSE
                    #     # model.saver.save(sess, sess_file) 

                    #     np.savez(train_predictions_file, data=train_pred)
                    #     np.savez(dev_predictions_file, data=pred)
                    #     np.savez(model_w_file, data=weight)
                    #     np.savez(model_b_file, data=bias)
                    #     np.savez(train_feature_file, data=train_features)
                    #     np.savez(dev_feature_file, data=dev_features)
                        
                    #     dev_R2_max = sklearn_r2

                    print('Dev set', 'dev ME', ME, 'dev RMSE',RMSE, 'dev R2',sklearn_r2,'dev RMSE_min',dev_RMSE_min,'dev R2 for min RMSE',dev_R2_max)
                    logging.info('Dev set dev ME %f dev RMSE %f dev R2 %f dev RMSE min %f dev_R2_for_min_RMSE %f',ME,RMSE,sklearn_r2,dev_RMSE_min,dev_R2_max)
                    
                    line_to_append = ['dev',str(ME), str(RMSE), str(sklearn_r2), str(dev_RMSE_min), str(dev_R2_max), str(correlation_coeff[0]), str(correlation_coeff[1])]
                    if found_min:
                        line_to_append.append('')
                        line_to_append.append('new dev RMSE min')

                    #worksheet_append_wrapper(worksheet, line_to_append)

                else:
                   print("Dev dataset not available. Skipping dev evaluation.") 

                if test_available:
                    #print scores on test set
                    test_pred = []
                    test_real = []
                    test_features = []
                    print(test_labels_file)
                    for batch in file_generator(test_data_file, test_labels_file, config.B, permuted_band = permuted_band):
                        pred_temp, feature = sess.run([model.pred, model.feature], feed_dict={
                            model.x: batch[0],
                            model.y: batch[1],
                            model.keep_prob: 1
                            })
                        test_pred.append(pred_temp)  
                        test_features.append(feature)
                    test_pred=np.concatenate(test_pred)
                    test_features = np.concatenate(test_features)
                    test_real = np.load(test_labels_file)['data']
                    test_pred = test_pred[:len(test_real)]
                    test_real = test_real[:len(test_pred)]
                    # if found_min:
                    #     test_pred_min = test_pred
                    #     test_real_min = test_real
                    print("test_real: ", np.round(test_real,2))
                    print("test_pred: ", np.round(test_pred,2))
                    # print("test_real_min: ", np.round(test_real_min,2))
                    # print("test_pred_min: ", np.round(test_pred_min,2))
                    if i<10000: test_real_writer.writerow(test_real)
                    test_pred_writer.writerow(test_pred)
                    print("test_error percentage: ", np.round(100*np.abs(test_real - test_pred) / ((test_real + test_pred) / 2),0 ))
                    print("mean test_error percentage: ", np.mean(100*np.abs(test_real - test_pred) / ((test_real + test_pred) / 2)))

                    print("test_diff: ",np.round(np.abs(test_real - test_pred),1 ))
                    print("mean test_diff: ", np.mean(np.abs(test_real - test_pred)))

                    RMSE=np.sqrt(np.mean((test_pred-test_real)**2))
                    rmse_test_save.append(RMSE)
                    if len(rmse_test_save) > 1000:
                        gradients = np.gradient(rmse_test_save[-500:])
                        if np.mean(gradients) > 0:
                            print('----------------------------------------------------------------')
                            print('----------------------------------------------------------------')
                            print('----------------------------------------------------------------')
                            print()
                            print("The gradient of RMSE_test is increasing over the last 500 points.")
                            print("The neural network is not learning effectively. Probably there is overfitting.")
                            print("Stopping..")

                    ME=np.mean(test_pred-test_real)
                    sklearn_r2 = r2_score(test_real, test_pred)
                    if len(test_pred)>1:
                        correlation_coeff = pearsonr(test_pred, test_real)
                    else:
                        correlation_coeff = [np.NaN,np.NaN]
                    
                    # if found_min:
                    #     np.savez(test_predictions_file, data=test_pred)
                    #     test_RMSE_min = RMSE
                    #     test_R2_max = sklearn_r2

                    if RMSE < test_RMSE_min:
                        print("Found a new test RMSE minimum")
                        test_RMSE_min = RMSE
                        model.saver.save(sess, sess_file)  # Save the TensorFlow model
                        # You might also want to save predictions, weights, features here, similar to what you do for the dev dataset
                        # For example:
                        np.savez(test_predictions_file, data=test_pred)
                        print()
                        np.savez(model_w_file, data=weight)
                        np.savez(model_b_file, data=bias)
                        np.savez(test_feature_file, data=test_features)

                    print('Test set', 'test ME', ME, 'test RMSE',RMSE, 'test R2',sklearn_r2,'test RMSE_min',test_RMSE_min,'test R2 for min RMSE',test_R2_max)
                    print
                    logging.info('Test set test ME %f test RMSE %f test R2 %f test RMSE min %f test_R2_for_min_RMSE %f',ME,RMSE,sklearn_r2,test_RMSE_min,test_R2_max)
                    
                    line_to_append = ['test',str(ME), str(RMSE), str(sklearn_r2), str(test_RMSE_min), str(test_R2_max), str(correlation_coeff[0]), str(correlation_coeff[1])]
                    #worksheet_append_wrapper(worksheet, line_to_append)

                
                else:
                    print("Test dataset not available. Skipping dev evaluation.") 

                summary_train_loss.append(str(train_loss))
                summary_RMSE.append(str(RMSE))
                summary_ME.append(str(ME))
                summary_R2.append(str(sklearn_r2))
           
            if restrict_iterations is not None and i == restrict_iterations:
                model.saver.save(sess, sess_file)
                return end_and_output_results(worksheet, train_RMSE_min, train_R2_max, dev_RMSE_min, dev_R2_max, test_RMSE_min, test_R2_max)

                
    except KeyboardInterrupt:
        print('stopped')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains neural network architectures.')
    parser.add_argument("-k", "--key_file", type=str, help="Path to the key file with input parameters.")

    args = parser.parse_args()

    if args.key_file:
            params = read_key_file(args.key_file)
            dataset_source_dir = params.get('DATASET_FOLDER')
            nnet_architecture = params.get('NNET_ARCHITECTURE')
            season_frac = float(params.get('SEASON_FRAC'))
            permuted_band = float(params.get('PERMUTED_BAND'))
            num_iters = int(params.get('NUM_ITERS'))
            load_weights = params.get('LOAD_WEIGHTS') == '1'
            weights_path_training = params.get('WEIGHTS_PATH_TRAINING')+"/weights/model.weights"
            data_augmentation = params.get('DATA_AUGMENTATION') == '1'
            yield_treshold = float(params.get('YIELD_TRESHOLD'))
            noise_level = float(params.get('NOISE_LEVEL'))
            replication_factor = float(params.get('REPLICATION_FACTOR'))

    else:
            print("Key file is required.")
            sys.exit(1)
    
    directory = dataset_source_dir
    if nnet_architecture == 'CNN':
        config = CNN_Config(season_frac)
        model= CNN_NeuralModel(config,'net')
        print("Running CNN...")
    elif nnet_architecture == 'LSTM':
        config = LSTM_Config(season_frac)
        model= LSTM_NeuralModel(config,'net')
        print("Running LSTM...")
    else:
        print('Error: did not specify a valid neural network architecture. Ending..')
        sys.exit()

    model.summary_op = tf.compat.v1.summary.merge_all()
    model.saver = tf.compat.v1.train.Saver()
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.22)
    
    # Launch the graph.
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.compat.v1.global_variables_initializer())
    if load_weights :
        model.saver.restore(sess, weights_path_training)
        print("Loaded weights from {}".format(weights_path_training))

    #INSERT NAME OF RESULTS GOOGLE SHEET HERE
    experiment_doc_name = ""
    run_NN(model, sess, directory, nnet_architecture, config, experiment_doc_name, restrict_iterations = num_iters, permuted_band = permuted_band)