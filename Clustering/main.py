from data import load_data
from models import create_tsne_model, create_kmeans_model
from plots import show_plots_by_tsne_model, shop_plots_by_kmeans_model


if __name__ == '__main__':
    X, y = load_data('../Exchange rates.csv')

    labels = {
        'x': 'Rating',
        'y': 'Season',
    }

    tsne = create_tsne_model(
        learning_rate='auto',
        n_iter=1000,
        init='pca',
        perplexity=30,
        angle=0.5,
    )
    tsne_transformed = tsne.fit_transform(X=X, y=y)

    kmeans = create_kmeans_model(
        n_clusters=8,
        n_init='auto',
        max_iter=1000,
        algorithm='elkan'
    )
    kmeans_transformed = kmeans.fit_transform(X=X, y=y)

    show_plots_by_tsne_model(tsne_transformed, labels, y.to_list())
    shop_plots_by_kmeans_model(kmeans_transformed, labels, y.to_list())
