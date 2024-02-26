from macpath import split
from operator import concat
import re
from cProfile import label
from cgi import test
from tkinter import Label
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import smiles2adjoin, molecular_fg
from rdkit import Chem
from random import Random
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from itertools import compress

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
         
num2str =  {i:j for j,i in str2num.items()}


class Graph_Classification_Dataset(object):  # Graph classification task data set processing
    def __init__(self,path,smiles_field1='Smiles1',smiles_field2='Smiles2',label_field=label, index_field=label, max_len=500,seed=1,batch_size=16,a=2,addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t',encoding='latin1')
        elif path.endswith('.xlsx'):
            self.df = pd.read_excel(path)
        else:
            self.df = pd.read_csv(path, encoding='latin1')
        self.smiles_field1 = smiles_field1
        self.smiles_field2 = smiles_field2
        self.label_field = label_field
        self.index_field = index_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field1].str.len() <= max_len]
        self.df = self.df[[True if Chem.MolFromSmiles(smi) is not None else False for smi in self.df[smiles_field1]]]
        self.seed = seed
        self.batch_size = batch_size
        self.a = a
        self.addH = addH

    def get_data(self):

        '''Randomized Split Dataset'''
        data = self.df
        data = data.fillna(666)
        train_idx = []
        idx = data.sample(frac=0.8).index
        train_idx.extend(idx)
        train_data = data[data.index.isin(train_idx)]
        data = data[~data.index.isin(train_idx)]
        test_idx = []
        idx = data[~data.index.isin(train_data)].sample(frac=0.5).index
        test_idx.extend(idx)
        test_data = data[data.index.isin(test_idx)]
        val_data = data[~data.index.isin(train_idx+test_idx)]
        df_train_data = pd.DataFrame(train_data)
        df_test_data = pd.DataFrame(test_data)
        df_val_data = pd.DataFrame(val_data)
        '''Splitting the dataset by random molecular scaffold, random_scaffold_split'''
        # data = self.df
        # data = data.fillna(666)
        # train_ids, val_ids, test_ids = random_scaffold_split(data, sizes=(0.8, 0.1, 0.1), balanced=True,seed=self.seed)
        # train_data = data.iloc[train_ids]
        # val_data = data.iloc[val_ids]
        # test_data = data.iloc[test_ids]
        # df_train_data = pd.DataFrame(train_data)
        # df_test_data = pd.DataFrame(test_data)
        # df_val_data = pd.DataFrame(val_data)

        '''Scaffold Split Dataset, scaffold_split'''
        # data1 = self.df1
        # smiles_list = data1[self.smiles_field1]    
        # df_train_data,df_val_data,df_test_data = scaffold_split(data1, smiles_list, task_idx=None, null_value=0,
        #            frac_train=0.8, frac_valid=0.1, frac_test=0.1,
        #            return_smiles=False)

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (df_train_data[self.smiles_field1], df_train_data[self.label_field], df_train_data[self.smiles_field2], df_train_data[self.index_field]))

        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(batch_size=self.batch_size, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]),tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).shuffle(1000).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((df_test_data[self.smiles_field1], df_test_data[self.label_field], df_test_data[self.smiles_field2], df_test_data[self.index_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]),tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((df_val_data[self.smiles_field1], df_val_data[self.label_field], df_val_data[self.smiles_field2], df_val_data[self.index_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]),tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)


        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        y = np.array(label).astype('int64')
        return x, adjoin_matrix, y

    def tf_numerical_smiles(self, smiles1, label, smiles2, index):
        x1,adjoin_matrix1,y= tf.py_function(self.numerical_smiles, [smiles1,label], [tf.int64, tf.float32 ,tf.int64])
        x1.set_shape([None])
        adjoin_matrix1.set_shape([None,None])
        y.set_shape([None])
        x2,adjoin_matrix2,index = tf.py_function(self.numerical_smiles, [smiles2,index], [tf.int64, tf.float32 ,tf.int64])
        x2.set_shape([None])
        adjoin_matrix2.set_shape([None,None])
        index.set_shape([None])
        return x1, adjoin_matrix1, y, x2,adjoin_matrix2, index


class Inference_Dataset(object):
    def __init__(self,sml_list,max_len=500,addH=True):
        self.vocab = str2num
        self.devocab = num2str
        self.sml_list = [i for i in sml_list if len(i)<max_len]
        self.addH =  addH

    def get_data(self):

        self.dataset = tf.data.Dataset.from_tensor_slices((self.sml_list,))
        self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]),tf.TensorShape([None]))).cache().prefetch(20)

        return self.dataset

    def numerical_smiles(self, smiles):
        smiles_origin = smiles
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        return x, adjoin_matrix,[smiles], atoms_list

    def tf_numerical_smiles(self, smiles):
        x,adjoin_matrix,smiles,atom_list = tf.py_function(self.numerical_smiles, [smiles], [tf.int64, tf.float32,tf.string, tf.string])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        return x, adjoin_matrix,smiles,atom_list
