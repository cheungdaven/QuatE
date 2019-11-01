class QuatE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(QuatE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 4 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def _calc(self, lhs, rel, rhs, forward=True):
        
        denominator = torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 + rel[3] ** 2)
        #print(denominator)
        rel_r = rel[0] #/ denominator
        rel_i = rel[1] #/ denominator
        rel_j = rel[2] #/ denominator
        rel_k = rel[3] #/ denominator
        

        A = lhs[0] * rel_r - lhs[1] * rel_i - lhs[2] * rel_j - lhs[3] * rel_k
        B = lhs[0] * rel_i + rel_r * lhs[1] + lhs[2] * rel_k - rel_j * lhs[3]
        C = lhs[0] * rel_j + rel_r * lhs[2] + lhs[3] * rel_i - rel_k * lhs[1]
        D = lhs[0] * rel_k + rel_r * lhs[3] + lhs[1] * rel_j - rel_i * lhs[2]
         
        if forward:
            score_r = A @ rhs[0].transpose(0, 1) + B @ rhs[1].transpose(0, 1) + C @ rhs[2].transpose(0, 1) + D @ rhs[3].transpose(0, 1)
        else:
            score_r = A * rhs[0] + B * rhs[1] + C * rhs[2] + D * rhs[3]
        return score_r

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:2*self.rank], lhs[:, 2*self.rank:3*self.rank], lhs[:, 3*self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank], rel[:, 2*self.rank:3*self.rank], rel[:, 3*self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:2*self.rank], rhs[:, 2*self.rank:3*self.rank], rhs[:, 3*self.rank:]

        return torch.sum(self._calc(lhs, rel, rhs, False),1, keepdim=True)

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:2*self.rank], lhs[:, 2*self.rank:3*self.rank], lhs[:, 3*self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank], rel[:, 2*self.rank:3*self.rank], rel[:, 3*self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:2*self.rank], rhs[:, 2*self.rank:3*self.rank], rhs[:, 3*self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:2*self.rank], to_score[:, 2*self.rank:3*self.rank], to_score[:, 3*self.rank:]

        
        score = self._calc(lhs, rel, to_score)
        return (score), (
                torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2 + lhs[2] ** 2 + lhs[3] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 + rel[3] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2 + rhs[2] ** 2 + rhs[3] ** 2))

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:2*self.rank], lhs[:, 2*self.rank:3*self.rank], lhs[:, 3*self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank], rel[:, 2*self.rank:3*self.rank], rel[:, 3*self.rank:]
    
        #denominator = torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 + rel[3] ** 2)
        
        rel_r = rel[0] #/ denominator
        rel_i = rel[1] #/ denominator
        rel_j = rel[2] #/ denominator
        rel_k = rel[3] #/ denominator
        
        A = lhs[0] * rel_r - lhs[1] * rel_i - lhs[2] * rel_j - lhs[3] * rel_k
        B = lhs[0] * rel_i + rel_r * lhs[1] + lhs[2] * rel_k - rel_j * lhs[3]
        C = lhs[0] * rel_j + rel_r * lhs[2] + lhs[3] * rel_i - rel_k * lhs[1]
        D = lhs[0] * rel_k + rel_r * lhs[3] + lhs[1] * rel_j - rel_i * lhs[2]

        return torch.cat([A, B, C, D], 1)
