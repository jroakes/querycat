# -*- coding: utf-8 -*-
import pandas as pd
import math
from mpmath import *
from collections import Counter
import json

from lib.normalization import normalize_corpus, tokenize_text
from lib.apriori import apriori




class Categorize:

    def __init__(self, df, col="query", **data):

        self.df = df
        self.counts = pd.DataFrame()
        self.categories = []
        self.col = col
        self.data = data
        self.categorize_queries(**data)


    @staticmethod
    def match_labels(x, labels):
        match = '##other##'
        for label in labels:
            if all(i in x.split() for i in label.split()):
                match = label
        return match



    def create_transactions(self, df, col):

        df[[col]]     = df[col].astype(str).dropna()
        queries       = df[col].tolist()

        # normalize corpus
        print("Normalizing the keyword corpus.")
        norm_queries = normalize_corpus(queries, lemmatize=True, only_text_chars=True, sort_text=True)
        match_queries = norm_queries
        norm_queries = list(set(norm_queries))

        print('Total queries:', len(match_queries))
        print('Total unique queries:', len(norm_queries))

        transactions = []

        for query in norm_queries:
            if len(query):
                transactions.append(query.split(' '))

        print('Total transactions:', len(transactions))

        return transactions, match_queries



    def categorize_queries(self, **data):

        """
        Executes Apriori algorithm and returns a RelationRecord generator.

        Arguments:
            transactions -- A transaction iterable object
                            (eg. [['A', 'B'], ['B', 'C']]).

        Keyword arguments:
            min_support -- The minimum support of relations (float).
            min_confidence -- The minimum confidence of relations (float).
            min_lift -- The minimum lift of relations (float).
            max_length -- The maximum length of the relation (integer).
        """

        min_support = data.get('min_support', 0.002)
        min_confidence = data.get('min_confidence', 0)
        min_lift = data.get('min_lift', 1)


        print("Converting to transactions.")
        transactions, match_queries = self.create_transactions(self.df, self.col)

        print("Running Apriori")
        results = list(apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, max_length=None))

        print("Making Categories")
        self.categories  = [' '.join(list(l.items)) for l in results]

        print('Total Categories: {}'.format(len(set(self.categories))))

        self.df['match_queries'] = match_queries
        self.df['category'] = self.df.match_queries.map(lambda x:  self.match_labels(x, self.categories))

        self.counts = pd.DataFrame(self.df.category.value_counts())
