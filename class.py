import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from dataset import Graph_Classification_Dataset
import os
import pandas as pd
from model import PredictModel, BertModel
from sklearn.metrics import roc_auc_score,confusion_matrix,precision_recall_curve,auc
from hyperopt import fmin, tpe, hp
from utils import get_task_names
from tensorflow.python.client import device_lib
from sklearn.preprocessing import StandardScaler
import pickle
import math
import csv

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def count_parameters(model):
    total_params = 0
    for variable in model.trainable_variables:
        shape = variable.shape
        params = 1
        for dim in shape:
            params *= dim
        total_params += params
    return total_params

def cover_dict(path):
    file_path = path
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    tensor_dict = {key: tf.constant(value) for key, value in data.items()}
    new_data = {i: value for i, (key, value) in enumerate(tensor_dict.items())}
    return new_data

def score(y_test, y_pred):
    auc_roc_score = roc_auc_score(y_test, y_pred)
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    PPV = tp / (tp + fp)
    NPV = tn / (fn + tn)
    return tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV

def DAR_feature(file_path, column_name):
    df = pd.read_excel(file_path)
    column_data = df[column_name].values.reshape(-1, 1)
    scaler = StandardScaler()
    column_data_standardized = scaler.fit_transform(column_data)
    column_data_normalized = tf.keras.utils.normalize(column_data_standardized, axis=0).flatten()
    data_dict = {index: tf.constant(value, dtype=tf.float32) for index, value in zip(df.index, column_data_normalized)}
    return data_dict

def process_list(input_list):
    input_list.append(np.mean(input_list))
    mean_value = np.mean(input_list[:-1])
    std_value = np.std(input_list[:-1], ddof=0)
    mean_range = f'{mean_value:.4f} ± {std_value:.4f}'
    input_list[-1] = mean_range
    print(input_list)
    return input_list
        
Heavy_dict = cover_dict('Heavy_1280.pkl')
Light_dict = cover_dict('Light_1280.pkl')
Antigen_dict = cover_dict('Antigen_1280.pkl')
DAR_dict = DAR_feature('data.xlsx', 'DAR_val')

def main(seed, args):

    task = 'ADC'
    idx = ['index']
    label = ['label（100nm）']

    arch = {'name': 'Medium', 'path': 'medium3_weights'}
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''
    trained_epoch = 20
    num_layers = 6
    d_model = 256
    addH = True
    dff = d_model * 2
    vocab_size = 18

    num_heads = args['num_heads']
    dense_dropout = args['dense_dropout']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    train_dataset, test_dataset, val_dataset = Graph_Classification_Dataset('data.xlsx', 
                                                                            smiles_field1='Payload Isosmiles',
                                                                            smiles_field2='Linker Isosmiles',
                                                                            label_field=label,
                                                                            index_field=idx, 
                                                                            seed=seed,
                                                                            batch_size=batch_size,
                                                                            a = len(label), 
                                                                            addH=addH).get_data()
                                                        
    x1, adjoin_matrix1, y, x2, adjoin_matrix2, index = next(iter(train_dataset.take(1)))

    seq1 = tf.cast(tf.math.equal(x1, 0), tf.float32)
    seq2 = tf.cast(tf.math.equal(x2, 0), tf.float32)
    mask1 = seq1[:, tf.newaxis, tf.newaxis, :]
    mask2 = seq2[:, tf.newaxis, tf.newaxis, :]

    heavy_tensor_list = []
    light_tensor_list = []
    antigen_tensor_list = []
    DAR_tensor_list = []
    for i in index.numpy():
        heavy_tensor_list.append(Heavy_dict[i[0]])
        light_tensor_list.append(Light_dict[i[0]])
        antigen_tensor_list.append(Antigen_dict[i[0]])
        DAR_tensor_list.append(DAR_dict[i[0]])
    t1 = np.vstack(heavy_tensor_list)
    t2 = np.vstack(light_tensor_list)
    t3 = np.vstack(antigen_tensor_list)
    t4 = np.vstack(DAR_tensor_list)
    
    model = PredictModel(num_layers=num_layers,
                         d_model=d_model,
                         dff=dff, 
                         num_heads=num_heads, 
                         vocab_size=vocab_size,
                         a=len(label),
                         dense_dropout = dense_dropout)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model,
                         dff=dff, num_heads=num_heads, vocab_size=vocab_size)

        pred = temp(x1, mask=mask1, training=True, adjoin_matrix=adjoin_matrix1)
        temp.load_weights(
            arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'], trained_epoch))
        temp.encoder.save_weights(
            arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        del temp

        pred = model(x1=x1, mask1=mask1, training=True, adjoin_matrix1=adjoin_matrix1, x2=x2,mask2=mask2, adjoin_matrix2=adjoin_matrix2, t1=t1,t2=t2,t3=t3,t4=t4)

        model.encoder.load_weights(
            arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        print('load_wieghts')

    total_params = count_parameters(model)

    print('*'*100)
    print("Total Parameters:", total_params)
    print('*'*100)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    auc = -10
    stopping_monitor = 0
    for epoch in range(200):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for x1, adjoin_matrix1, y, x2, adjoin_matrix2, index in train_dataset:
            heavy_tensor_list = []
            light_tensor_list = []
            antigen_tensor_list = []
            DAR_tensor_list = []
            for i in index.numpy():
                heavy_tensor_list.append(Heavy_dict[i[0]])
                light_tensor_list.append(Light_dict[i[0]])
                antigen_tensor_list.append(Antigen_dict[i[0]])
                DAR_tensor_list.append(DAR_dict[i[0]])
            t1 = np.vstack(heavy_tensor_list)
            t2 = np.vstack(light_tensor_list)
            t3 = np.vstack(antigen_tensor_list)
            t4 = np.vstack(DAR_tensor_list)
            with tf.GradientTape() as tape:
                seq1 = tf.cast(tf.math.equal(x1, 0), tf.float32)
                mask1 = seq1[:, tf.newaxis, tf.newaxis, :]
                seq2 = tf.cast(tf.math.equal(x2, 0), tf.float32)
                mask2 = seq2[:, tf.newaxis, tf.newaxis, :]
                preds = model(x1=x1, mask1=mask1,training=True,adjoin_matrix1=adjoin_matrix1, x2=x2, mask2=mask2, adjoin_matrix2=adjoin_matrix2,t1=t1,t2=t2,t3=t3,t4=t4)
                loss = loss_object(y,preds)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print('epoch: ', epoch, 'loss: {:.4f}'.format(loss.numpy().item()))

        y_true = []
        y_preds = []
        for x1, adjoin_matrix1, y, x2, adjoin_matrix2, index in val_dataset:
            heavy_tensor_list = []
            light_tensor_list = []
            antigen_tensor_list = []
            DAR_tensor_list = []
            for i in index.numpy():
                heavy_tensor_list.append(Heavy_dict[i[0]])
                light_tensor_list.append(Light_dict[i[0]])
                antigen_tensor_list.append(Antigen_dict[i[0]])
                DAR_tensor_list.append(DAR_dict[i[0]])
            t1 = np.vstack(heavy_tensor_list)
            t2 = np.vstack(light_tensor_list)
            t3 = np.vstack(antigen_tensor_list)
            t4 = np.vstack(DAR_tensor_list)
            seq1 = tf.cast(tf.math.equal(x1, 0), tf.float32)
            mask1 = seq1[:, tf.newaxis, tf.newaxis, :]
            seq2 = tf.cast(tf.math.equal(x2, 0), tf.float32)
            mask2 = seq2[:, tf.newaxis, tf.newaxis, :]
            preds = model(x1=x1, mask1=mask1,training=False,adjoin_matrix1=adjoin_matrix1, x2=x2, mask2=mask2, adjoin_matrix2=adjoin_matrix2,t1=t1,t2=t2,t3=t3,t4=t4)
            y_label = y
            y_pred = preds
            y_true.append(y_label)
            y_preds.append(y_pred)
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true,y_preds)

        print('val auc:{:.4f}'.format(auc_new))
        if auc_new> auc:
            auc = auc_new
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch, trained_epoch, pretraining_str),
                    [y_true, y_preds])
            model.save_weights('classification_weights/{}_{}.h5'.format(task, seed))
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        if stopping_monitor > 30:
            break

    y_true = []
    y_preds = []
    model.load_weights('classification_weights/{}_{}.h5'.format(task, seed))
    for x1, adjoin_matrix1, y, x2, adjoin_matrix2, index in test_dataset:
        heavy_tensor_list = []
        light_tensor_list = []
        antigen_tensor_list = []
        DAR_tensor_list = []
        for i in index.numpy():
            heavy_tensor_list.append(Heavy_dict[i[0]])
            light_tensor_list.append(Light_dict[i[0]])
            antigen_tensor_list.append(Antigen_dict[i[0]])
            DAR_tensor_list.append(DAR_dict[i[0]])
        t1 = np.vstack(heavy_tensor_list)
        t2 = np.vstack(light_tensor_list)
        t3 = np.vstack(antigen_tensor_list)
        t4 = np.vstack(DAR_tensor_list)
        seq1 = tf.cast(tf.math.equal(x1, 0), tf.float32)
        mask1 = seq1[:, tf.newaxis, tf.newaxis, :]
        seq2 = tf.cast(tf.math.equal(x2, 0), tf.float32)
        mask2 = seq2[:, tf.newaxis, tf.newaxis, :]
        preds = model(x1=x1, mask1=mask1,training=False,adjoin_matrix1=adjoin_matrix1, x2=x2, mask2=mask2, adjoin_matrix2=adjoin_matrix2,t1=t1,t2=t2,t3=t3,t4=t4)
        y_label = y
        y_pred = preds
        y_true.append(y_label)
        y_preds.append(y_pred)

    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    y_preds = tf.sigmoid(y_preds).numpy()
    test_auc = roc_auc_score(y_true, y_preds)
    
    tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = score(y_true, y_preds)
    print('test auc:{:.4f}'.format(test_auc))

    return test_auc,tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV                                                     

space = {"dense_dropout": hp.quniform("dense_dropout", 0, 0.5, 0.05), 
        "learning_rate": hp.loguniform("learning_rate", np.log(3e-5), np.log(15e-5)),
        "batch_size": hp.choice("batch_size", [16,32,48,64]),
        "num_heads": hp.choice("num_heads", [4,8]),
        }

def hy_main(args):
    test_auc_list = []
    x = 0
    for seed in [2, 8, 9]:
        print(seed)
        test_auc,tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = main(seed, args)
        test_auc_list.append(test_auc)
        x+= test_auc
    test_auc_list.append(np.mean(test_auc_list))
    print(test_auc_list)
    print(args["dense_dropout"])
    print(args["learning_rate"])
    print(args["batch_size"])
    print(args["num_heads"])
    return -x/3

best = fmin(hy_main, space, algo = tpe.suggest, max_evals= 30)
print(best)
best_dict = {}
a = [16,32,48,64]
b = [4, 8]
best_dict["dense_dropout"] = best["dense_dropout"]
best_dict["learning_rate"] = best["learning_rate"]
best_dict["batch_size"] = a[best["batch_size"]]
best_dict["num_heads"] = b[best["num_heads"]]
print(best_dict)

if __name__ == '__main__':
    test_auc_list = []
    tp_l, tn_l, fn_l, fp_l, se_l, sp_l, mcc_l, acc_l, auc_roc_score_l, F1_l, BA_l, prauc_l, PPV_l, NPV_l = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
    lists_to_process = [tp_l, tn_l, fn_l, fp_l, se_l, sp_l, mcc_l, acc_l, auc_roc_score_l, F1_l, BA_l, prauc_l, PPV_l, NPV_l]
    for seed in [2,8,9]:
        print(seed)
        test_auc,tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = main(seed, best_dict)
        test_auc_list.append(test_auc)
        tp_l.append(tp)
        tn_l.append(tn)
        fn_l.append(fn)
        fp_l.append(fp)
        se_l.append(se)
        sp_l.append(sp)
        mcc_l.append(mcc)
        acc_l.append(acc)
        auc_roc_score_l.append(auc_roc_score)
        F1_l.append(F1)
        BA_l.append(BA)
        prauc_l.append(prauc)
        PPV_l.append(PPV)
        NPV_l.append(NPV)
    test_auc_list.append(np.mean(test_auc_list))
    tp_l.append(np.mean(tp_l))
    tn_l.append(np.mean(tn_l))
    fn_l.append(np.mean(fn_l))
    fp_l.append(np.mean(fp_l))
    se_l.append(np.mean(se_l))
    sp_l.append(np.mean(sp_l))
    mcc_l.append(np.mean(mcc_l))
    acc_l.append(np.mean(acc_l))
    auc_roc_score_l.append(np.mean(auc_roc_score_l))
    F1_l.append(np.mean(F1_l))
    BA_l.append(np.mean(BA_l))
    prauc_l.append(np.mean(prauc_l))
    PPV_l.append(np.mean(PPV_l))
    NPV_l.append(np.mean(NPV_l))
    
    for i in range(len(lists_to_process)):
        lists_to_process[i] = process_list(lists_to_process[i])
    filename = 'ADCNet_output.csv'
    column_names = ['tp', 'tn', 'fn', 'fp', 'se', 'sp', 'mcc', 'acc', 'auc', 'F1', 'BA', 'prauc','PPV', 'NPV']
    rows = zip(tp_l, tn_l, fn_l, fp_l, se_l, sp_l, mcc_l, acc_l, auc_roc_score_l, F1_l, BA_l, prauc_l, PPV_l, NPV_l)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(rows)
    print(f'CSV file {filename} was successfully written')


