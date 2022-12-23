from matplotlib import pyplot as plt
import plotly.express as px

from Clustering.constants import WINTER, SPRING, SUMMER, AUTUMN, MARKER_SIZE


def show_plots_by_tsne_model(corr, labels, color):
    fig = px.scatter(None, x=corr[:, 0], y=corr[:, 1], labels=labels, opacity=1, color=color)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                     showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                     showline=True, linewidth=1, linecolor='black')

    fig.update_layout(plot_bgcolor='white')
    fig.update_traces(marker=dict(size=MARKER_SIZE))

    fig.write_html('TSNE.html')


def shop_plots_by_kmeans_model(corr, labels, color):
    plt.xlabel(labels['x'], fontsize=14, fontweight="bold")
    plt.ylabel(labels['y'], fontsize=14, fontweight="bold")

    def season_to_color(value):
        if value == WINTER:
            return 'b'
        elif value == SPRING:
            return 'g'
        elif value == SUMMER:
            return 'm'
        elif value == AUTUMN:
            return 'y'

    color = [season_to_color(item) for item in color]

    plt.scatter(corr[:, 0], corr[:, 1], c=color, s=MARKER_SIZE)
    plt.show()
