import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset
from tdc.multi_pred import DrugRes


def smiles2morgan(s, radius=2, nBits=1024):
    """
    Converts SMILES to morgan fingerprint using utils from rdkit
    :param s: SMILES string repr of the chemical compound
    :param radius:
    :param nBits: the size of the morgan vector repr
    :return:
    """
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
        features = np.zeros((nBits,))
    return features


def test_smiles2morgan():
    for smiles in ["OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1",  # glucose
                   "CN=C=O"]:  # methyl isocyanate
        for nbits in [256, 512, 1024]:
            fingerprint = smiles2morgan(smiles, nBits=nbits)
            assert fingerprint.shape[0] == nbits
            assert np.all(np.logical_or(fingerprint == 0, fingerprint == 1))
    print("smiles2morgan - All tests passed.")


class GDSC1Dataset(Dataset):
    def __init__(self, df, fingerprinting=None, transform=None, target_transform=None):
        self.df = df
        self.fingerprinting = fingerprinting
        self.transform = transform
        self.target_transform = target_transform
        pass

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        drug_fingerprint = self.df.iloc[idx, 1]
        gene_expr = self.df.iloc[idx, 3]
        ic50 = self.df.iloc[idx, 4]
        if self.fingerprinting:
            drug_fingerprint = self.fingerprinting(drug_fingerprint)
        if self.transform:
            gene_expr = self.transform(gene_expr)
        if self.target_transform:
            ic50 = self.target_transform(ic50)
        return (drug_fingerprint, gene_expr), ic50


def get_GDSC1_splits(batch_size=32, fingerprinting=smiles2morgan):
    """
    :param batch_size: for the dataloader
    :param fingerprinting: mapping that converts smiles string to a feature vector
    :return: a dict containing dataloaders for train, test, and validation data.
    """
    data = DrugRes(name='GDSC1')
    split = data.get_split()
    dataloaders = {}
    for portion in ['train', 'test']:
        df = split[portion]
        column_normalize(df, col_name='Cell Line')
        dataset = GDSC1Dataset(df, fingerprinting=fingerprinting)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders[portion] = dataloader
    dataloaders['test'] = split['test']
    return dataloaders

def column_normalize(df, col_name='Cell Line'):
    """
    Performs in-place normalization on a column of a dataframe so that the \
    values are scaled to between 0 and 1. Column data type can be int, float, or
    numpy.array

    :param df: Dataframe to perform the normalization on
    :param col_name: Name of the column in the dataframe
    :return: N/A
    """
    if not col_name in df.columns:
        print("Please choose a column that's included in the dataframe!")
        return
    M = np.max(df[col_name].apply(np.max))
    m = np.min(df[col_name].apply(np.min))
    df[col_name] = df[col_name].apply(lambda x: (x-m)/(M-m))

if __name__ == "__main__":
    pass
    # test_smiles2morgan()
