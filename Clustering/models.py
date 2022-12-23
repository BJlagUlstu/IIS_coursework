from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def create_tsne_model(learning_rate, n_iter, init, perplexity, angle):
    return TSNE(
        learning_rate=learning_rate,
        n_iter=n_iter,
        init=init,
        perplexity=perplexity,
        angle=angle,
    )


def create_kmeans_model(n_clusters, n_init, max_iter, algorithm):
    return KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algorithm,
    )
