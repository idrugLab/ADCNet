import torch
import esm
import pandas as pd
import pickle

# Load ESM-2 model
# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)

df = pd.read_excel(r'data.xlsx',)

protain = df['Antigen Sequence（64）'].tolist()
smiles = df['ADC ID'].tolist()

datas = []
for i in range(len(protein)):
    datas.append((smiles[i], protain[i])) 

sequence_representations = []

for data in datas:
    # print(data)
    try :
        batch_labels, batch_strs, batch_tokens = batch_converter([data])
    except Exception as e:
        print(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.

    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
esm_embedding = {}
for i in range(len(sequence_representations)):
    esm_embedding[datas[i][0]] = sequence_representations[i]
print(sequence_representations[0].shape)
print(len(esm_embedding))

file_path = 'Antigen.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(esm_embedding, file)

# 加载保存的长字典
with open(file_path, 'rb') as file:
    loaded_dict = pickle.load(file)
    
# Look at the unsupervised self-attention map contact predictions
