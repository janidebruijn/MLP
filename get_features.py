def get_features(self, text_list, ngram=1, max_features=5000):
    if self.counter is None:
        self.counter = CountVectorizer(ngram_range=(1, ngram), max_features=max_features)
        features_matrix = self.counter.fit_transform(text_list)
    else:
        features_matrix = self.counter.transform(text_list)

    return features_matrix.toarray()