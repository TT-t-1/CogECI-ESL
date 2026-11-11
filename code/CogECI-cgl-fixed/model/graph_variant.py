"""Disclosed graph-operator substitution for Table 5; no author implementation was provided."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, HypergraphConv, TransformerConv
from torch_geometric.nn.dense import Linear
from model.CGE import group

class GraphVariant(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.ModuleDict({'event': Linear(768, args.n_last)})
        self.kind = args.graph_encoder
        self.beta_intra = args.beta_intra
        self.beta_inter = args.beta_inter
        self.convs = nn.ModuleDict()
        for relation in ['intra', 'inter']:
            if self.kind == 'gcn':
                conv = GCNConv(args.n_last, args.n_last, add_self_loops=False)
            elif self.kind == 'hypergraph':
                conv = HypergraphConv(args.n_last, args.n_last)
            else:
                conv = TransformerConv(args.n_last, args.n_last // args.num_heads,
                                       heads=args.num_heads, root_weight=False)
            self.convs[relation] = conv

    def forward(self, nodes, edges):
        x = self.proj['event'](nodes['event'])
        outputs = []
        for (_, kind, _), index in edges.items():
            if self.kind == 'hypergraph':
                # Each directed edge becomes a two-node hyperedge; hypergraphs are undirected.
                count = index.shape[1]
                index = torch.stack((index.t().reshape(-1),
                                     torch.arange(count, device=x.device).repeat_interleave(2)))
            outputs.append(F.relu(self.convs[kind](x, index)))
        return {'event': outputs[0] if len(outputs) == 1 else group(outputs, self.beta_intra, self.beta_inter)}
