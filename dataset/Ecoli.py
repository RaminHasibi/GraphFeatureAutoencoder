from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.utils import dense_to_sparse, to_undirected, remove_self_loops
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os.path as osp

class Ecoli_Exp(InMemoryDataset):
    def __init__(self, root, network='TF_net', Normalize=False, transform=None, pre_transform=None):
        self.network = network
        self.normalize = Normalize
        super(InMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return []


    @property
    def processed_file_names(self):
        return ['processed_{}_data.pt'.format(self.network)]

    def download(self):
        pass

    def index_to_mask(self, index, size):
        mask = torch.zeros((size,), dtype=torch.bool)
        mask[index] = 1
        return mask

    def read_TF_net(self, root):
        TF_gene = pd.read_csv(root + "/network_tf_gene.txt", skiprows=34, header=None, usecols=[0, 1, 2, 4],
                              delimiter='\t')
        TF_gene = TF_gene.apply(lambda x: x.astype(str).str.lower())
        TF_gene = TF_gene[((TF_gene[2] == '-') | (TF_gene[2] == '+'))]
        TF_gene = TF_gene.drop_duplicates(subset=[0, 1])

        Ecoli = pd.read_table(root + '/avg_E_coli_v4_Build_6_exps466probes4297.tab')

        Ecoli['E_coli_v4_Build_6:genes'] = Ecoli['E_coli_v4_Build_6:genes'].str.split('_').str[0]
        Ecoli = Ecoli.apply(lambda x: x.astype(str).str.lower())
        Ecoli = Ecoli.rename(columns={"E_coli_v4_Build_6:genes": "Genes"})

        tf_regdb = TF_gene[0].isin(Ecoli.Genes) & TF_gene[1].isin(Ecoli.Genes)
        Filt_regDB = TF_gene[tf_regdb]
        Filt_DB_genes = np.union1d(Filt_regDB[0].unique(), Filt_regDB[1].unique())
        Ecoli_Filt = Ecoli[Ecoli.Genes.isin(Filt_DB_genes)]

        Adj = np.zeros([len(Filt_DB_genes), len(Filt_DB_genes)])
        features = np.zeros([len(Filt_DB_genes), Ecoli.shape[1] - 1])
        for i in range(len(Filt_regDB)):
            row = np.where(Filt_DB_genes == Filt_regDB.iloc[i][0])[0][0]
            col = np.where(Filt_DB_genes == Filt_regDB.iloc[i][1])[0][0]
            if (Filt_regDB.iloc[i][2] == '+'):
                Adj[row][col] = 1
                Adj[col][row] = 1
            else:
                Adj[row][col] = 1
                Adj[row][col] = 1

        print(len(Adj.nonzero()[0]))
        for i in range(len(Filt_DB_genes)):
            features[i] = Ecoli[Ecoli.Genes == Filt_DB_genes[i]].iloc[:, 1:]

        return dense_to_sparse(torch.tensor(Adj))[0], torch.tensor(features, dtype=torch.float32)

    def read_PPI(self, root):
        BioGrid = pd.read_csv(root + '/BIOGRID-ORGANISM-Escherichia_coli_K12_W3110-3.5.180.tab2.txt', delimiter='\t')
        BioGrid['Official Symbol Interactor A'] = BioGrid['Official Symbol Interactor A'].str.lower()
        BioGrid['Official Symbol Interactor B'] = BioGrid['Official Symbol Interactor B'].str.lower()
        BioGrid = BioGrid.rename(
            columns={"Official Symbol Interactor A": "Gene_A", "Official Symbol Interactor B": "Gene_B"})

        Ecoli = pd.read_table(root + '/avg_E_coli_v4_Build_6_exps466probes4297.tab')

        Ecoli['E_coli_v4_Build_6:genes'] = Ecoli['E_coli_v4_Build_6:genes'].str.split('_').str[0]
        Ecoli = Ecoli.apply(lambda x: x.astype(str).str.lower())
        Ecoli = Ecoli.rename(columns={"E_coli_v4_Build_6:genes": "Genes"})

        Filt_BioGrid_indeces = BioGrid.Gene_A.isin(Ecoli.Genes) & BioGrid.Gene_B.isin(
            Ecoli.Genes)  # & BioGrid['Experimental System Name'] != 'Biochemical Activity'
        Filt_BioGrid = BioGrid[Filt_BioGrid_indeces]
        Filt_BioGrid_PPI = Filt_BioGrid[Filt_BioGrid['Experimental System Type'] == 'physical']
        Filt_BioGrid_PPI_Genes = np.union1d(Filt_BioGrid_PPI.Gene_A.unique(), Filt_BioGrid_PPI.Gene_B.unique())
        Ecoli_Filt_PPI = Ecoli[Ecoli.Genes.isin(Filt_BioGrid_PPI_Genes)]

        Adj = np.zeros([len(Filt_BioGrid_PPI_Genes), len(Filt_BioGrid_PPI_Genes)])
        features = np.zeros([len(Filt_BioGrid_PPI_Genes), Ecoli_Filt_PPI.shape[1] - 1])
        for i in range(len(Filt_BioGrid_PPI)):
            row = np.where(Filt_BioGrid_PPI_Genes == Filt_BioGrid_PPI.iloc[i][7])[0][0]
            col = np.where(Filt_BioGrid_PPI_Genes == Filt_BioGrid_PPI.iloc[i][8])[0][0]
            Adj[row][col] = 1
            Adj[col][row] = 1

        for i in range(len(Filt_BioGrid_PPI_Genes)):
            features[i] = Ecoli[Ecoli.Genes == Filt_BioGrid_PPI_Genes[i]].iloc[:, 1:]

        return dense_to_sparse(torch.tensor(Adj))[0], torch.tensor(features, dtype=torch.float32)

    def read_Genetic(self, root):
        BioGrid = pd.read_csv(root + '/BIOGRID-ORGANISM-Escherichia_coli_K12_W3110-3.5.180.tab2.txt', delimiter='\t')
        BioGrid['Official Symbol Interactor A'] = BioGrid['Official Symbol Interactor A'].str.lower()
        BioGrid['Official Symbol Interactor B'] = BioGrid['Official Symbol Interactor B'].str.lower()
        BioGrid = BioGrid.rename(
            columns={"Official Symbol Interactor A": "Gene_A", "Official Symbol Interactor B": "Gene_B"})

        Ecoli = pd.read_table(root + '/avg_E_coli_v4_Build_6_exps466probes4297.tab')

        Ecoli['E_coli_v4_Build_6:genes'] = Ecoli['E_coli_v4_Build_6:genes'].str.split('_').str[0]
        Ecoli = Ecoli.apply(lambda x: x.astype(str).str.lower())
        Ecoli = Ecoli.rename(columns={"E_coli_v4_Build_6:genes": "Genes"})

        Filt_BioGrid_indeces = BioGrid.Gene_A.isin(Ecoli.Genes) & BioGrid.Gene_B.isin(
            Ecoli.Genes)  # & BioGrid['Experimental System Name'] != 'Biochemical Activity'
        Filt_BioGrid = BioGrid[Filt_BioGrid_indeces]
        Filt_BioGrid_Genetic = Filt_BioGrid[Filt_BioGrid['Experimental System Type'] == 'genetic']
        Filt_BioGrid_Genetic_Genes = np.union1d(Filt_BioGrid_Genetic.Gene_A.unique(),
                                                Filt_BioGrid_Genetic.Gene_B.unique())
        Ecoli_Filt_Genetic = Ecoli[Ecoli.Genes.isin(Filt_BioGrid_Genetic_Genes)]

        Adj = np.zeros([len(Filt_BioGrid_Genetic_Genes), len(Filt_BioGrid_Genetic_Genes)])
        features = np.zeros([len(Filt_BioGrid_Genetic_Genes), Ecoli_Filt_Genetic.shape[1] - 1])
        for i in range(len(Filt_BioGrid_Genetic)):
            row = np.where(Filt_BioGrid_Genetic_Genes == Filt_BioGrid_Genetic.iloc[i][7])[0][0]
            col = np.where(Filt_BioGrid_Genetic_Genes == Filt_BioGrid_Genetic.iloc[i][8])[0][0]
            Adj[row][col] = 1
            Adj[col][row] = 1

        for i in range(len(Filt_BioGrid_Genetic_Genes)):
            features[i] = Ecoli[Ecoli.Genes == Filt_BioGrid_Genetic_Genes[i]].iloc[:, 1:]

        return dense_to_sparse(torch.tensor(Adj))[0], torch.tensor(features, dtype=torch.float32)


    def process(self):
        if self.network == 'TF_net':
            edge_index, x = self.read_TF_net(self.root)
        elif self.network == 'PPI':
            edge_index, x = self.read_PPI(self.root)
        elif self.network == 'Genetic':
            edge_index, x = self.read_Genetic(self.root)
        indeces = np.arange(len(x))
        train_indeces, test_indeces = train_test_split(indeces, test_size=.33, random_state=42)
        # train_indeces,valid_indeces = train_test_split(train_valid_indeces,test_size=.15,random_state=42)

        train_feats_indeces, test_feats_indeces = train_test_split(np.arange(x.size(1)), test_size=.33, random_state=42)
        y = x[:, test_feats_indeces]
        x = x[:, train_feats_indeces]
        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]
        train_mask = self.index_to_mask(train_indeces, x.size(0))
        # val_mask = self.index_to_mask(valid_indeces, x.size(0))
        test_mask = self.index_to_mask(test_indeces, x.size(0))

        #         if self.normalize:
        #             scaler = StandardScaler()
        #             scaler.fit(x)
        #             scaler.tra
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        # data.val_mask = val_mask
        data.test_mask = test_mask
        torch.save(self.collate([data]), self.processed_paths[0])
