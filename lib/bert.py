import torch
import pandas as pd
from torch import nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import transformers
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import random
import matplotlib.pyplot as plt

from querycat import config

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

class BERTSim:

    def __init__(self, dims = None):

        if 'distilbert' in config.TRANSFORMER_MODEL:
            self.model      = DistilBertModel.from_pretrained(config.TRANSFORMER_MODEL)
            self.tokenizer  = DistilBertTokenizer.from_pretrained(config.TRANSFORMER_MODEL)
        else:
            self.model      = BertModel.from_pretrained(config.TRANSFORMER_MODEL)
            self.tokenizer  = BertTokenizer.from_pretrained(config.TRANSFORMER_MODEL)

        self.terms          = []
        self.embeddings     = torch.FloatTensor([])
        self.embeddings_2d  = None
        self.diffs          = []
        self.embed          = nn.Linear(self.model.config.dim, dims) if dims else None
        self.sim_fn         = torch.nn.CosineSimilarity(dim=1)


    def read_df(self,df, term_col = 'terms', diff_col = 'diffs'):
        self.add_terms(df[term_col].tolist())
        self.diffs = df[diff_col].tolist()


    def add_terms(self, texts):
        for t in texts:
            if t not in self.terms:
                emb   = self.get_embedding(t)
                self.terms.append(t)
                self.embeddings = torch.cat((self.embeddings, emb), dim=0)


    def get_embedding(self, text):
        with torch.no_grad():
            input_ids = torch.LongTensor(self.tokenizer.encode(text, add_special_tokens=False)).unsqueeze(0)
            outputs = self.model(input_ids)
            lh = outputs[0]
            if self.embed is not None:
                lh = self.embed(lh)
            emb = torch.sum(lh, dim=1)

        return emb


    def get_most_similar(self, term):
        emb = self.get_embedding(term)
        comp = emb.repeat(len(self.embeddings), 1)
        sim = self.sim_fn(self.embeddings, comp)
        best = sim.argmax().item()
        return self.terms[best], sim[best].item()


    def get_similar_df(self, term):
        emb = self.get_embedding(term)
        comp = emb.repeat(len(self.embeddings), 1)
        sim = self.sim_fn(self.embeddings, comp)
        df = pd.DataFrame(columns=['terms', 'sim'])
        df['terms'] = self.terms
        df['sim']   = sim.tolist()
        df.sort_values(by='sim', ascending=False, inplace=True)
        return df


    def convert_tsne(self):
        tsne_model = TSNE(perplexity=60, n_components=2, n_iter=2000, random_state=config.RANDOM_SEED)
        self.embeddings_2d = tsne_model.fit_transform(self.embeddings)

    def convert_pca(self):
        self.embeddings_2d = PCA(n_components=2).fit_transform(self.embeddings)

    def convert_umap(self):
        self.embeddings_2d = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=config.RANDOM_SEED).fit_transform(self.embeddings)


    def diff_plot(self, reduction='tsne'):

        if reduction == 'tsne':
            self.convert_tsne()
        elif reduction == 'pca':
            self.convert_pca()
        elif reduction == 'umap':
            self.convert_umap()
        else:
            raise Exception("`reduction` must be `tsne`,`pca`, or `umap`.")


        x_coords = []
        y_coords = []
        for value in self.embeddings_2d:
            x_coords.append(value[0])
            y_coords.append(value[1])

        plt.figure(figsize=(24, 16))

        for i,term in enumerate(self.terms):
            x = x_coords[i]
            y = y_coords[i]
            d = float(self.diffs[i])
            if d >= 0:
              plt.scatter(x, y, color='green', s=abs(d*200))
              plt.text(x+0.03, y+0.03, term, fontsize=10)
            else:
              plt.scatter(x, y, color='red', s=abs(d*200))
              plt.text(x+0.03, y+0.03, term, fontsize=10)

        plt.show()
