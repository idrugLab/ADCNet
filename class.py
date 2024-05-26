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
    sp = tn / (tn + fp)  # 也是R
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
def run_experiment(seed_list, best_dict):
    results = {
        'test_auc': [], 'tp': [], 'tn': [], 'fn': [], 'fp': [],
        'se': [], 'sp': [], 'mcc': [], 'acc': [],
        'auc_roc_score': [], 'F1': [], 'BA': [],
        'prauc': [], 'PPV': [], 'NPV': []
    }

    for seed in seed_list:
        print(seed)
        result_values = main(seed, best_dict)
        for key, value in zip(results.keys(), result_values):
            results[key].append(value)

    for key in results:
        results[key].append(np.mean(results[key]))
    
    return results

def save_results_to_csv(results, filename):
    column_names = ['tp', 'tn', 'fn', 'fp', 'se', 'sp', 'mcc', 'acc', 'auc', 'F1', 'BA', 'prauc', 'PPV', 'NPV']
    rows = zip(results['tp'], results['tn'], results['fn'], results['fp'], results['se'], results['sp'], 
               results['mcc'], results['acc'], results['auc_roc_score'], results['F1'], results['BA'], 
               results['prauc'], results['PPV'], results['NPV'])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(rows)
        
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
                loss = 0
                for i in range(len(label)):
                    y_label = y[:,i]
                    y_pred = preds[:,i]
                    validId = np.where((y_label == 0) | (y_label == 1))[0]
                    if len(validId) == 0:
                        continue
                    y_t = tf.gather(y_label,validId)
                    y_p = tf.gather(y_pred,validId)
            
                    loss += loss_object(y_t, y_p)
                loss = loss/(len(label))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print('epoch: ', epoch, 'loss: {:.4f}'.format(loss.numpy().item()))

        y_true = {}
        y_preds = {}
        for i in range(len(label)):
            y_true[i] = []
            y_preds[i] = []
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
            for i in range(len(label)):
                y_label = y[:,i]
                y_pred = preds[:,i]
                y_true[i].append(y_label)
                y_preds[i].append(y_pred)
        y_tr_dict = {}
        y_pr_dict = {}
        for i in range(len(label)):
            y_tr = np.array([])
            y_pr = np.array([])
            for j in range(len(y_true[i])):
                a = np.array(y_true[i][j])
                b = np.array(y_preds[i][j])
                y_tr = np.concatenate((y_tr,a))
                y_pr = np.concatenate((y_pr,b))
            y_tr_dict[i] = y_tr
            y_pr_dict[i] = y_pr

        AUC_list = []

        for i in range(len(label)):
            y_label = y_tr_dict[i]
            y_pred = y_pr_dict[i]
            validId = np.where((y_label== 0) | (y_label == 1))[0]
            if len(validId) == 0:
                continue
            y_t = tf.gather(y_label,validId)
            y_p = tf.gather(y_pred,validId)
            if all(target == 0 for target in y_t) or all(target == 1 for target in y_t):
                AUC = float('nan')
                AUC_list.append(AUC)
                continue
            y_p = tf.sigmoid(y_p).numpy()
            AUC_new = sklearn.metrics.roc_auc_score(y_t, y_p, average=None)

            AUC_list.append(AUC_new)
        auc_new = np.nanmean(AUC_list)

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

    y_true = {}
    y_preds = {}
    for i in range(len(label)):
        y_true[i] = []
        y_preds[i] = []
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
        for i in range(len(label)):
            y_label = y[:,i]
            y_pred = preds[:,i]
            y_true[i].append(y_label)
            y_preds[i].append(y_pred)
    y_tr_dict = {}
    y_pr_dict = {}
    for i in range(len(label)):
        y_tr = np.array([])
        y_pr = np.array([])
        for j in range(len(y_true[i])):
            a = np.array(y_true[i][j])
            if a.ndim == 0:
                continue
            b = np.array(y_preds[i][j])
            y_tr = np.concatenate((y_tr,a))
            y_pr = np.concatenate((y_pr,b))
        y_tr_dict[i] = y_tr
        y_pr_dict[i] = y_pr
    auc_list = []
    for i in range(len(label)):
        y_label = y_tr_dict[i]
        y_pred = y_pr_dict[i]
        validId = np.where((y_label== 0) | (y_label == 1))[0]
        if len(validId) == 0:
            continue
        y_t = tf.gather(y_label,validId)
        y_p = tf.gather(y_pred,validId)
        if all(target == 0 for target in y_t) or all(target == 1 for target in y_t):
            AUC = float('nan')
            auc_list.append(AUC)
            continue
        y_p = tf.sigmoid(y_p).numpy()
        AUC_new = sklearn.metrics.roc_auc_score(y_t, y_p, average=None)
        auc_list.append(AUC_new)

    test_auc = np.nanmean(auc_list)
    tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = score(y_t, y_p)
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
    for seed in [9]:
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
    return -x

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
    seed_list = [2, 8, 9]
    results = run_experiment(seed_list, best_dict)
    filename = 'FG-BERT_output.csv'
    save_results_to_csv(results, filename)
        writer.writerow(column_names)
        writer.writerows(rows)
    print(f'CSV 文件 {filename} 写入完成')


