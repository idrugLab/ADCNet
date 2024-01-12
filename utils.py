from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import numpy as np
import openbabel as ob
import os
import csv
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog

def fg_list(): # 47 FGs list
    fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1,6,fName)
    fg_list = []
    for i in range(fparams.GetNumFuncGroups()):
        fg_list.append(fparams.GetFuncGroup(i))
    fg_list.pop(27)
    
    x = [Chem.MolToSmiles(_) for _ in fg_list]+['*C=C','*F','*Cl','*Br','*I','[Na+]','*P','*P=O','*[Se]','*[Si]']
    y = set(x)
    return list(y)

def obsmitosmile(smi):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def molecular_fg(smiles): # Getting functional groups (including rings) in molecules

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '
    a = fg_list()


    ssr = Chem.GetSymmSSSR(mol)
    num_ring = len(ssr)
    ring_dict = {}
    for i in range(num_ring):
        ring_dict[i+1] = list(ssr[i])

    f_g_list = []
    for i in ring_dict.values():
        f_g_list.append(i)

    for i in a:
        patt = Chem.MolFromSmarts(i)
        flag = mol.HasSubstructMatch(patt)
        if flag:
            atomids = mol.GetSubstructMatches(patt)
            for atomid in atomids:
                f_g_list.append(list(atomid))
    return f_g_list


def smiles2adjoin(smiles,explicit_hydrogens=True,canonical_atom_order=False): # Converting molecules in SMILES format to atom lists and adjacency matrices

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()

    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())


    adjoin_matrix = np.eye(num_atoms)
    num_bonds = mol.GetNumBonds()
    
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u,v] = 1.0
        adjoin_matrix[v,u] = 1.0
    
    return atoms_list,adjoin_matrix


def get_header(path):
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(path, use_compound_names=False):
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names