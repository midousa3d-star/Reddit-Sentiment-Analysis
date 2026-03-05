# streamlit run app/gui.py

import streamlit as st
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from reddit_scraping import subreddit_posts_scrape, subreddit_comments_scrape
from sentiment_analysis import sentiment_results_df
from dataframe_operations import merge_data_and_results, save_df_to_csv
from data_visualisation import create_violin_plot, combined_plot, plot_correlation
from data_visualisation import sentiment_bar_plot, word_list_to_upvote_correlation, text_clustering, text_clustering_dbscan
from wordcloud_generator import generate_word_cloud_df

# from process_input import on_scrape_button_click

# Selecting inputs
st.title('Reddit Sentiment Analysis')

subreddit_name = st.text_input("Subreddit name (eg: news, aww, datascience)", key="subreddit_name")
subreddit_name = subreddit_name.lower()

# Create columns for title, post type, time inputs
col1a, col2a, col3a = st.columns(3)

with col1a:
    data_to_scrape = st.selectbox('What to scrape?', ('Post Titles', 'Post Comments'))

with col2a:
    data_filter = st.selectbox('Select post type', ('Top', 'Hot'))

time_filter = None
if data_filter == 'Top':
    with col3a:
        time_filter = st.selectbox('Filter by time', ("Hour", "Day", "Week", "Month", "Year", "All"))
data_filter = data_filter.lower()
if time_filter:
    time_filter = time_filter.lower()

# Create columns for slider and button
col1b, col2b = st.columns([7, 1])

with col1b:
    number_of_posts = st.slider('Select maximum number of posts', 1, 1000, value=5)
    # st.write("Maximum number of posts:", number_of_posts)

with col2b:
    st.markdown(
        """
        <style>
        .centered-button {
            display: flex;
            align-items: center;
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    scrape_button_clicked = st.button("Analyse")
    st.markdown('</div>', unsafe_allow_html=True)

# Place progress bar and dynamic text underneath all input elements
if scrape_button_clicked:
    try:
        progress_text = st.empty()  # Placeholder for dynamic text
        progress_bar = st.progress(0)

        post_num_stages = 4
        post_progress_unit = math.floor(100 / post_num_stages)

        # Scraping Posts
        if data_to_scrape == 'Post Titles':
            progress_text.text("Scraping data...")
            data_df = subreddit_posts_scrape(subreddit_name, data_filter, time_filter, number_of_posts)
            results_df = sentiment_results_df(data_df)
            st.session_state["data_df"] = data_df
            st.session_state["results_df"] = results_df

            # Visualise results (combined plot should be split and all two plots should be in two columns)

            # violin plot, sentiment bar plot row
            progress_bar.progress(post_progress_unit)
            # Create columns for violin and bar chart
            col1c, col2c = st.columns(2)
            with col1c:
                progress_text.text("Plotting results...")
                fig = create_violin_plot(results_df, 
                                        title=f'r/{subreddit_name} {data_filter} posts sentiment analysis', 
                                        bgcolor='white', 
                                        filename=f'{subreddit_name}_{data_filter}_{number_of_posts}_posts_violin_chart')
                st.subheader("Violin Plot")
                st.pyplot(fig)
            with col2c:
                progress_text.text("Generating sentiment bar chart...")
                st.subheader('Sentiment Bar Chart')
                fig = sentiment_bar_plot(results_df)
                st.pyplot(fig)

            # Wordcloud generator
            progress_bar.progress(2*post_progress_unit)
            progress_text.text("Generating wordcloud...")
            st.subheader('Wordcloud')
            # Display the word cloud
            fig, ax = plt.subplots()
            wordcloud = generate_word_cloud_df(data_df) # ends in plt.show
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            # Text clustering, example sentences from cluster
            progress_bar.progress(3*post_progress_unit)
            progress_text.text("Clustering sentences...")
            st.subheader('Sentence clustering')
            # Display
            fig,kmeans, corpus, corpus_embeddings = text_clustering(data_df)
            st.pyplot(fig)

            # Displaying the examples from each cluster
            st.markdown("### Examples of sentences from the clusters")

            text_and_cluster_label = pd.concat([pd.Series(kmeans.labels_), pd.DataFrame(corpus)], axis=1)
            text_and_cluster_label.columns = ['cluster', 'text']


            for i in range(text_and_cluster_label['cluster'].nunique()):
                st.markdown(f"##### Cluster {i}")
                # st.write(f'Center of the cluster: {}')
                example_sentences = text_and_cluster_label[text_and_cluster_label['cluster']==i].head(5)
                st.table(example_sentences)
            
            progress_text.text("Done!")
            progress_bar.progress(100)

        # Scraping comments
        elif data_to_scrape == 'Post Comments':
            
            progress_text.text("Scraping data...")
            data_df = subreddit_comments_scrape(subreddit_name, data_filter, time_filter, number_of_posts)
            results_df = sentiment_results_df(data_df)
            st.session_state["data_df"] = data_df
            st.session_state["results_df"] = results_df

            # Visualise results (combined plot should be split and all two plots should be in two columns)

            # violin plot, sentiment bar plot row
            progress_bar.progress(post_progress_unit)
            # Create columns for violin and bar chart
            col1c, col2c = st.columns(2)
            with col1c:
                progress_text.text("Plotting results...")
                fig = create_violin_plot(results_df, 
                                        title=f'r/{subreddit_name} {data_filter} posts sentiment analysis', 
                                        bgcolor='white', 
                                        filename=f'{subreddit_name}_{data_filter}_{number_of_posts}_posts_violin_chart')
                st.subheader("Violin Plot")
                st.pyplot(fig)
            with col2c:
                progress_text.text("Generating sentiment bar chart...")
                st.subheader('Sentiment Bar Chart')
                fig = sentiment_bar_plot(results_df)
                st.pyplot(fig)

            # Wordcloud generator
            progress_bar.progress(2*post_progress_unit)
            progress_text.text("Generating wordcloud...")
            st.subheader('Wordcloud')
            # Display the word cloud
            fig, ax = plt.subplots()
            wordcloud = generate_word_cloud_df(data_df) # ends in plt.show
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            # Text clustering, example sentences from cluster
            progress_bar.progress(3*post_progress_unit)
            progress_text.text("Clustering sentences...")
            st.subheader('Sentence clustering')
            # Display
            fig, model, corpus, corpus_embeddings = text_clustering(data_df)
            st.pyplot(fig)

            # Displaying the examples from each cluster
            st.markdown("### Examples of sentences from the clusters")

            text_and_cluster_label = pd.concat([pd.Series(model.labels_), pd.DataFrame(corpus)], axis=1)
            text_and_cluster_label.columns = ['cluster', 'text']


            for i in range(text_and_cluster_label['cluster'].nunique()):
                st.markdown(f"##### Cluster {i}")
                # st.write(f'Center of the cluster: {}')
                example_sentences = text_and_cluster_label[text_and_cluster_label['cluster']==i].head(5)
                st.table(example_sentences)
            
            progress_text.text("Done!")
            progress_bar.progress(100)


    except Exception as e:
        error_message = st.empty()
        st.markdown(
            """
            <style>
            .fade-out {
                opacity: 1;
                transition: opacity 0.1s ease-in-out;
            }
            .fade-out.hidden {
                opacity: 0;
                transition: opacity 0.1s ease-in-out;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="fade-out hidden">', unsafe_allow_html=True)
        error_message.error(f"An error occurred: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        # time.sleep(5)
        # error_message.empty()
