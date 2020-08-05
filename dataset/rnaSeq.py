from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.utils import dense_to_sparse, to_undirected, remove_self_loops
import torch
import numpy as np
from sklearn.model_selection import train_test_split



class RnaSeq(InMemoryDataset):

    def __init__(self, root, network='HumanPPI', transform=None, pre_transform=None):
        self.network = network
        super(InMemoryDataset, self).__init__(root, transform, pre_transform)
        if self.network == 'HumanPPI':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.network == 'SmallPPI':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.network == 'MousePPI':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['HumanPPI_processed_rnaSeq_data.pt', 'SmallPPI_processed_rnaSeq_data.pt',
                'MousePPI_processed_rnaSeq_data.pt']

    def download(self):
        pass

    def index_to_mask(self, indices, index, shape):
        mask = torch.zeros(shape)
        mask[indices[0, index], indices[1, index]] = 1
        return mask

    def process(self):
        edge_index = torch.tensor(np.load(self.root + '/' + self.network + '.npy'))
        x = np.load(self.root + '/' + 'rnaSeq.npy', allow_pickle=True)
        gene_names = x[:, 0]
        x = torch.tensor(np.array(x[:, 1:], dtype=np.float), dtype=torch.float)
        print(x.size(0))
        matrix_mask = torch.zeros([x.size(0), x.size(1)])
        matrix_mask[x.nonzero(as_tuple=True)] = 1
        indices = np.array(x.data.numpy().nonzero())
        print(indices)
        ix_train, ix_test = train_test_split(np.arange(len(indices[0])), test_size=0.25, random_state=42)
        print(ix_train)
        # ix_train, ix_val = train_test_split(ix_train_val, test_size=.2, random_state=42)
        train_mask = self.index_to_mask(indices, ix_train, [x.size(0), x.size(1)])
        # val_mask = self.index_to_mask(matrix_mask,[ ix_val, ix_val], [x.size(0),x.size(1)])
        test_mask = self.index_to_mask(indices, ix_test, [x.size(0), x.size(1)])

        #         edge_index = to_undirected(edge_index)
        #         edge_index = remove_self_loops(edge_index)[0]

        #                 if self.normalize:
        #                     scaler = StandardScaler()
        #                      scaler.fit(x)
        #                     scaler.tra
        data = Data(x=x, edge_index=edge_index, y=x)
        data.train_mask = train_mask
        #         data.val_mask = val_mask
        data.test_mask = test_mask
        data.gene_names = gene_names
        if self.network == 'HumanPPI':
            torch.save(self.collate([data]), self.processed_paths[0])
        elif self.network == 'SmallPPI':
            torch.save(self.collate([data]), self.processed_paths[1])
        elif self.network == 'MousePPI':
            torch.save(self.collate([data]), self.processed_paths[2])