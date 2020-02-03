# -*- coding: utf-8 -*-
import pandas as pd
import math
from mpmath import *
from collections import Counter
import json

from .normalization import normalize_corpus, tokenize_text
from .apriori import apriori
import pyfpgrowth as pg




class Categorize:

    def __init__(self, df, col="query", alg="apriori", **data):

        self.df = df
        self.counts = pd.DataFrame()
        self.categories = []
        self.col = col
        self.alg = alg
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
            min_lift -- The minimum lift of relations (float). (>1 is likely)
            min_probability -- Finds patterns that are associated with another with a certain minimum probability:

            min_confidence -- The minimum confidence of relations (float).


        """

        min_support = data.get('min_support', 10)
        min_lift = data.get('min_lift', 1)
        min_probability = data.get('min_probability', 0.5)
        min_confidence = data.get('min_confidence', 0)


        print("Converting to transactions.")
        transactions, match_queries = self.create_transactions(self.df, self.col)

        if self.alg.lower() == "apriori":
            print("Running Apriori")
            min_support = float(min_support/len(transactions))
            results = list(apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, max_length=None))
            print("Making Categories")
            self.categories  = [' '.join(list(l.items)) for l in results]

        elif self.alg.lower() == "fpgrowth":
            print("Running FPGrpwth")
            results = list(pg.generate_association_rules(pg.find_frequent_patterns(transactions, min_support), min_probability))
            print("Making Categories")
            self.categories  = [' '.join(l) for l in results]

        else:
            raise Exception("{} is not one of the available algorithms (`apriori`, `fpgrowth`)".format(self.alg))


        print('Total Categories: {}'.format(len(set(self.categories))))

        self.df['match_queries'] = match_queries
        self.df['category'] = self.df.match_queries.map(lambda x:  self.match_labels(x, self.categories))

        self.counts = pd.DataFrame(self.df.category.value_counts())
