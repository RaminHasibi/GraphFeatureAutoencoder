from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.utils import dense_to_sparse, to_undirected, remove_self_loops
import torch
import numpy as np
from sklearn.model_selection import train_test_split



class RnaSeq(InMemoryDataset):

    def __init__(self, root, network='MousePPI', transform=None, pre_transform=None):
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
        edge_index = torch.tensor(np.array(
            np.load(self.root + '/' + self.network + '.npy', allow_pickle=True), dtype=np.int)).t()
        gene_names = None
        if self.network == 'MousePPI':
            x = torch.tensor(np.load(self.root + '/' + 'mouse_rnaSeq.npy', allow_pickle=True), dtype=torch.float)
        else:
            x = np.load(self.root + '/' + 'rnaSeq.npy', allow_pickle=True)
            gene_names = x[:, 0]
            x = torch.tensor(np.array(x[:, 1:], dtype=np.float), dtype=torch.float)
        print(x.size(0))
        matrix_mask = torch.zeros([x.size(0), x.size(1)])
        matrix_mask[x.nonzero(as_tuple=True)] = 1
        indices = np.array(x.data.numpy().nonzero())
        ix_train, ix_test = train_test_split(np.arange(len(indices[0])), test_size=0.25, random_state=42)

        data = Data(x=x, edge_index=edge_index, y=x, nonzeromask=matrix_mask)
        data.gene_names = gene_names if gene_names is not None else None
        if self.network == 'HumanPPI':
            torch.save(self.collate([data]), self.processed_paths[0])
        elif self.network == 'SmallPPI':
            torch.save(self.collate([data]), self.processed_paths[1])
        elif self.network == 'MousePPI':
            torch.save(self.collate([data]), self.processed_paths[2])