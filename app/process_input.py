import streamlit as st
import time
import math

from reddit_scraping import subreddit_posts_scrape
from sentiment_analysis import sentiment_results_df
from dataframe_operations import merge_data_and_results, save_df_to_csv
from data_visualisation import create_violin_plot, combined_plot

post_n_stages = 3
post_progress_unit = math.floor(100/post_n_stages)

# Processing inputs

def on_scrape_button_click(subreddit_name, data_to_scrape, data_filter, time_filter, number_of_posts, progress_text, progress_bar):


    # Scraping Posts

    if data_to_scrape == 'Post Titles':

        # scrape_posts_and_plot_sentiment(subreddit_name='india', data_filter='top', time='year', number_of_posts=100)
        
        # scrape data
        progress_text.text(f"Scraping data...")
        data_df = subreddit_posts_scrape(subreddit_name, data_filter, time_filter, number_of_posts)
        progress_bar.progress(post_progress_unit)


        # generate sentiment df
        progress_text.text(f"Calculating sentiment...")
        results_df = sentiment_results_df(data_df)
        # make a combined df
        df = merge_data_and_results(data_df, results_df)
        progress_bar.progress(post_progress_unit*2)
        

        # # save the result and data csv 
        # save_df_to_csv(df, filepath=f'csv_files\post_data\{subreddit_name}_{data_filter}_{number_of_posts}_posts_data.csv')
        # save_df_to_csv(results_df, filepath=f'csv_files\post_result\{subreddit_name}_{data_filter}_{number_of_posts}_posts_results.csv')


        # visualise results
        progress_text.text(f"Plotting results...")
        create_violin_plot(df, 
                        title=f'r/{subreddit_name} {data_filter} posts sentiment analysis', 
                        bgcolor='white', 
                        filename=f'{subreddit_name}_{data_filter}_{number_of_posts}_posts_violin_chart')
        combined_plot(df, title=f'r/{subreddit_name} {data_filter} posts sentiment analysis',
                    filename=f'{subreddit_name}_{data_filter}_posts_combined_plots.png')
        progress_bar.progress(100)
        

    # Scraping comments
    elif data_to_scrape == 'Post Comments':
        pass
    




