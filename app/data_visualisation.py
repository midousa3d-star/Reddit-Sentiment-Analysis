import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd


#create histograms of neg, neu and pos sentiments given dataframe and title of the histogram

def create_histograms(dataframe, title):

    # Create histograms
    ax = dataframe[['roberta_neg', 'roberta_neu', 'roberta_pos']].hist(bins=10, figsize=(20, 8))
    # ax = df[['roberta_neg', 'roberta_pos']].hist(bins=20, figsize=(20, 10))

    # Determine the global x and y limits
    x_min = min([a.get_xlim()[0] for row in ax for a in row])
    x_max = max([a.get_xlim()[1] for row in ax for a in row])
    y_min = min([a.get_ylim()[0] for row in ax for a in row])
    y_max = max([a.get_ylim()[1] for row in ax for a in row])

    # Set the same x and y limits for all histograms
    for row in ax:
        for a in row:
            a.set_xlim(x_min, x_max)
            a.set_ylim(y_min, y_max)

    # Title
    # fig.suptitle(title, fontsize=16)

    plt.show()

# create_histograms(df)


# create violin plot given a df with 'roberta_pos', neu and neg given dataframe and title

def create_violin_plot(dataframe, title='Sentiment Analysis', bgcolor='white', filename='subreddit top'):

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(1, 3, figsize=(8, 5)) # 12, 5

    # Set the background color for the figure and axes
    fig.patch.set_facecolor(bgcolor)
    for ax in axes:
        ax.set_facecolor(bgcolor)

    # Plot each violin plot with fixed y-axis limits
    sns.violinplot(data=dataframe, y='roberta_pos', ax=axes[0], color='lime')
    axes[0].set_title('Positive sentiment')
    axes[0].set_ylabel('')
    axes[0].set_ylim(0, 1)  # Fixed y-axis limits

    sns.violinplot(data=dataframe, y='roberta_neu', ax=axes[1], color='gray')
    axes[1].set_title('Neutral sentiment')
    axes[1].set_ylabel('')
    axes[1].set_ylim(0, 1)  # Fixed y-axis limits

    sns.violinplot(data=dataframe, y='roberta_neg', ax=axes[2], color='red')
    axes[2].set_title('Negative sentiment')
    axes[2].set_ylabel('')
    axes[2].set_ylim(0, 1)  # Fixed y-axis limits

    # Title
    fig.suptitle(f'{title}', fontsize=16)

    # Adjust layout for better spacing
    plt.tight_layout()

    return fig


# plots correlation between all numerical values

def plot_correlation(dataframe, title='Correlation chart', bgcolor='white', save_fig=False, combined_plot=False, ax_of_combined_plot=None):

    if combined_plot == False:
        # Create a figure and axes for the subplots
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        ax=ax_of_combined_plot
        fig=plt.figure()

    # Set the background color for the figure and axes
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    # select only numerical values of this dataframe
    num_df = dataframe.select_dtypes(include=['number'])
    
    # plotting correlation heatmap 
    sns.heatmap(num_df.corr(), cmap="viridis", annot=True, ax=ax) 

    # Set the title for the heatmap
    # ax.set_title(title, fontsize=16)

    # if combined_plot!=True:
    #     # displaying heatmap 
    #     plt.show() 

    return fig


# sentiment bar chart plotting

def sentiment_bar_plot(df, title='Sentiment bar chart', bgcolor='white', combined_plot=False, ax_of_combined_plot=None):

    if combined_plot == False:
        # Create a figure and axes for the subplots
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        ax=ax_of_combined_plot
        fig=plt.figure()

    # Set the background color for the figure and axes
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    sentiment_counts = df['sentiment'].value_counts()

    # Define the colors for each sentiment
    colors = sentiment_counts.index.map({
        'positive': 'green',
        'neutral': 'gray',
        'negative': 'red'
    })

    # Plot the bar chart with custom colors
    sentiment_counts.plot(kind='bar', color=colors, ax=ax)

    # Add labels and title for better readability
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=16)

    # if combined_plot!=True:
    #     plt.show()

    return fig


# combined plots (for posts)

def combined_plot(df, title='combined plot', filename='combined_plot'):
    
    # Create a single figure to hold all three plots
    # fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Set the background color for the figure and axes
    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')

    # Adjust layout for better spacing
    fig.subplots_adjust(wspace=0.2)
    
    # Plot each chart in a subplot
    # create_violin_plot(df, title=f'Violin plot of sentiments', combined_plot=True, ax_of_combined_plot=axs[0])
    plot_correlation(df, title=f'Correlation among features', combined_plot=True, ax_of_combined_plot=axs[0])
    sentiment_bar_plot(df, title=f'Sentiment bar chart', combined_plot=True, ax_of_combined_plot=axs[1])

    # Set the main title for the combined figure
    fig.suptitle(title, fontsize=20)

    # Create directory if it does not exist
    output_dir = 'visualisation_images'
    os.makedirs(output_dir, exist_ok=True)
    # Save the combined figure
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path)

    # Show the combined figure
    # plt.show()

    return fig


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import nltk
from collections import Counter
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

# Function to count the most used words
def count_most_used_words(df):
    all_words = []
    stop_words = set(stopwords.words('english'))

    for text in df['text']:
        tokens = word_tokenize(text.lower())  # Convert to lowercase to count words case-insensitively
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        all_words.extend(filtered_tokens)

    # Count the occurrences of each word
    word_counts = Counter(all_words)

    # Convert the Counter to a DataFrame
    word_counts_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])
    word_counts_df = word_counts_df.sort_values(by='Count', ascending=False)

    return word_counts_df


# # only one word is allowed in the word list
# def word_list_to_upvote_correlation(df):

#     word_counts_df = count_most_used_words(df)
#     word_list = list(word_counts_df.head(10)['Word'])

#     # lower casing the word list
#     word_list_lower = []
#     for word in word_list:
#         word_list_lower.append(word.lower())

#     # list of dictionaries to make list word count dataframe
#     word_count_info_list = []

#     # go through all texts
#     for text in df['text']:
#         # convert to lower case and tokenise
#         tokens = word_tokenize(text.lower())
#         # count the number of times the words in word list as appeared
#         word_count = pd.DataFrame(tokens).value_counts().reset_index()
#         word_count.columns = ['Words', 'Count']

#         word_count_info = {}
#         # iterate through the words in the given list
#         for list_word in word_list_lower:
#             # if the chosen word is in the word_count df
#             for word in word_count['Words']:
#                 if word == list_word:
#                     # add the the count to the dictionary by selecting count value such that the 
#                     word_count_info[f'{list_word} count'] = word_count['Count'][word_count['Words']==list_word]
#         word_count_info_list.append(word_count_info)

#     # make a new df with columns for the count of each of those words in the word list. column name = [word]count
#     word_count_info_df = pd.DataFrame(word_count_info_list)

#     # take the score column and join it with this word count dataframe
#     score_and_word_count = pd.concat([df['score'], word_count_info_df], axis=1)

#     score_text_and_word_count = pd.concat([df['score'], df['text'], word_count_info_df], axis=1)

#     fig = plot_correlation(word_count_info_df)

#     return fig


def word_list_to_upvote_correlation(df):

    word_counts_df = count_most_used_words(df)
    word_list = list(word_counts_df.head(5)['Word'])

    # lower casing the word list
    word_list_lower = []
    for word in word_list:
        word_list_lower.append(word.lower())

    # list of dictionaries to make list word count dataframe
    word_count_info_list = []

    # go through all texts
    for text in df['text']:
        # convert to lower case and tokenise
        tokens = word_tokenize(text.lower())
        # count the number of times the words in word list as appeared
        word_count = pd.DataFrame(tokens).value_counts().reset_index()
        word_count.columns = ['Words', 'Count']

        word_count_info = {}
        # iterate through the words in the given list
        for list_word in word_list_lower:
            # if the chosen word is in the word_count df
            for word in word_count['Words']:
                if word == list_word:
                    # add the the count to the dictionary by selecting count value such that the 
                    word_count_info[f'{list_word} count'] = word_count['Count'][word_count['Words']==list_word]
        word_count_info_list.append(word_count_info)

    # make a new df with columns for the count of each of those words in the word list. column name = [word]count
    word_count_info_df = pd.DataFrame(word_count_info_list)

    # take the score column and join it with this word count dataframe
    score_and_word_count = pd.concat([df['score'], word_count_info_df], axis=1)

    score_text_and_word_count = pd.concat([df['score'], df['text'], word_count_info_df], axis=1)

    # plot correlation
    def plot_correlation(df):
        fig = plt.figure(figsize=(16, 6))
        heatmap = sns.heatmap(df.corr().fillna(0), vmin=-1, vmax=1, annot=True)
        # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
        heatmap.set_title('Correlation between most common words and upvotes', fontdict={'fontsize':12}, pad=12)
        return fig

    fig = plot_correlation(score_and_word_count)
    return fig


# text clustering

import pandas as pd, numpy as np
import torch, os, scipy
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer

def find_optimal_k(corpus_embeddings, max_k=10):
    """
    Finds the optimal number of clusters for KMeans using the Elbow Method and Silhouette Score.
    
    Parameters:
    corpus_embeddings: ndarray
        The embeddings of the corpus to be clustered.
    max_k: int
        The maximum number of clusters to test.
        
    Returns:
    best_k: int
        The optimal number of clusters.
    """
    sse = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(corpus_embeddings)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(corpus_embeddings, kmeans.labels_))

    # Finding the best K using silhouette scores
    best_k = range(2, max_k + 1)[silhouette_scores.index(max(silhouette_scores))]
    return best_k

def text_clustering(df):

    dataset = df 
    # Shuffling the dataset and taking the first 10000 rows
    corpus = list(dataset.sample(frac=1, random_state=42)[:10000]['text'])

    # Using SentenceTransformer
    model_path = "paraphrase-distilroberta-base-v1"
    model = SentenceTransformer(model_path)

    # Encoding the corpus
    corpus_embeddings = model.encode(corpus)
    corpus_embeddings.shape

    best_k = find_optimal_k(corpus_embeddings)

    kmeans = KMeans(n_clusters=best_k,random_state=0).fit(corpus_embeddings)
    cls_dist = pd.Series(kmeans.labels_).value_counts()

    # Finding one real sentence embedding, closest to each centroid point
    distances = scipy.spatial.distance.cdist(kmeans.cluster_centers_,corpus_embeddings)
    centers={}
    # print("Cluster", "Size", "Center-idx","Center-Example", sep="\t\t")
    # centerslist = []
    for i,d in enumerate(distances):
        ind = np.argsort(d, axis=0)[0]
        centers[i]=ind
        # print(i,cls_dist[i], ind, corpus[ind] ,sep="\t\t")
        # centers.append(corpus[ind])

    text_and_cluster_label = pd.concat([pd.Series(kmeans.labels_), pd.DataFrame(corpus)], axis=1)
    text_and_cluster_label.columns = ['cluster', 'text']

    X = umap.UMAP(n_components=2,min_dist=0.0).fit_transform(corpus_embeddings)
    labels= kmeans.labels_
    # print(labels)

    fig, ax = plt.subplots(figsize=(12,12))
    # print(X[:,0])
    plt.scatter(X[:,0], X[:,1], c=labels, s=10, cmap='Paired')
    for c in centers:
        plt.text(X[centers[c],0], X[centers[c], 1],"Cluster "+ str(c), fontsize=18)
    # plt.colorbar()

    return fig, kmeans, corpus, corpus_embeddings



def text_clustering_dbscan(df, eps=0.5, min_samples=5):
    """
    Clusters text data using DBSCAN and visualizes the clusters with UMAP.

    Parameters:
    df: DataFrame
        The DataFrame containing the text data.
    eps: float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples: int
        The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    fig: Figure
        The matplotlib figure object of the cluster visualization.
    dbscan: DBSCAN
        The DBSCAN clustering model.
    corpus: list
        The list of text data used for clustering.
    corpus_embeddings: ndarray
        The embeddings of the text data.
    """
    
    # Shuffling the dataset and taking the first 10000 rows
    corpus = list(df.sample(frac=1, random_state=42)[:10000]['text'])

    # Using SentenceTransformer
    model_path = "paraphrase-distilroberta-base-v1"
    model = SentenceTransformer(model_path)

    # Encoding the corpus
    corpus_embeddings = model.encode(corpus)

    # Applying DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(corpus_embeddings)
    labels = dbscan.labels_
    cls_dist = pd.Series(labels).value_counts()

    # Finding one real sentence embedding, closest to each centroid point
    unique_labels = set(labels)
    centers = {}
    for label in unique_labels:
        if label != -1:
            cluster_indices = np.where(labels == label)[0]
            cluster_embeddings = corpus_embeddings[cluster_indices]
            centroid = cluster_embeddings.mean(axis=0)
            distances = scipy.spatial.distance.cdist([centroid], cluster_embeddings)
            closest_index = cluster_indices[np.argmin(distances)]
            centers[label] = closest_index

    text_and_cluster_label = pd.concat([pd.Series(labels), pd.DataFrame(corpus)], axis=1)
    text_and_cluster_label.columns = ['cluster', 'text']

    X = umap.UMAP(n_components=2, min_dist=0.0).fit_transform(corpus_embeddings)

    fig, ax = plt.subplots(figsize=(12,12))
    plt.scatter(X[:,0], X[:,1], c=labels, s=10, cmap='Paired')
    for label, center in centers.items():
        plt.text(X[center, 0], X[center, 1], "Cluster " + str(label), fontsize=18)
    
    return fig, dbscan, corpus, corpus_embeddings