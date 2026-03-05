from reddit_scraping import subreddit_posts_scrape, subreddit_comments_scrape
from sentiment_analysis import sentiment_results_df, sentiment_results_all_comments
from dataframe_operations import merge_data_and_results, save_df_to_csv, use_csv_for_dataframe
from data_visualisation import create_violin_plot, combined_plot, create_histograms, sentiment_bar_plot, plot_correlation


# Packaging the code for posts #

# scrape a subreddit from scratch, save the csvs and create visualisation.

def scrape_posts_and_plot_sentiment(subreddit_name, data_filter, time, number_of_posts):
    
    global data_df, df

    # scrape data
    data_df = subreddit_posts_scrape(subreddit_name, data_filter, time, number_of_posts)
    # generate result df
    results_df = sentiment_results_df(data_df)
    # make a combined df
    df = merge_data_and_results(data_df, results_df)

    # save the result and data csv 
    save_df_to_csv(df, filepath=f'csv_files\post_data\{subreddit_name}_{data_filter}_{number_of_posts}_posts_data.csv')
    save_df_to_csv(results_df, filepath=f'csv_files\post_result\{subreddit_name}_{data_filter}_{number_of_posts}_posts_results.csv')

    # display a combined plot of violin, correlation and bar chart
    create_violin_plot(df, 
                       title=f'r/{subreddit_name} {data_filter} posts sentiment analysis', 
                       bgcolor='white', 
                       filename=f'{subreddit_name}_{data_filter}_{number_of_posts}_posts_violin_chart')
    combined_plot(df, title=f'r/{subreddit_name} {data_filter} posts sentiment analysis',
                  filename=f'{subreddit_name}_{data_filter}_posts_combined_plots.png')
    

# just scrape (not analysing sentiment) and put the data in a folder

def scrape_post_data_and_save(subreddit_name, data_filter, time, number_of_posts):

    # scrape data
    data_df = subreddit_posts_scrape(subreddit_name, data_filter, time, number_of_posts)

    # save the result and data csv 
    save_df_to_csv(data_df, filepath=f'csv_files\post_data\{subreddit_name}_{data_filter}_{number_of_posts}_posts_data.csv')


# use an already saved data

def open_data_csv_and_plot_sentiment(subreddit_name, scrape_data_type):

    results_or_data = 'data'
    filepath = f"csv_files/{results_or_data}/{subreddit_name}_{scrape_data_type}_{results_or_data}.csv"

    data_df = use_csv_for_dataframe(filepath)
    results_df = sentiment_results_df(data_df)
    df = merge_data_and_results(data_df, results_df)

    save_df_to_csv(df, subreddit_name, "top1000_post_titles", "results")
    # create_histograms(df, subreddit_name)
    create_violin_plot(df, subreddit_name)


# use an already saved result

def open_result_csv_and_plot_sentiment(subreddit_name, scrape_data_type):

    results_or_data = 'results'
    filepath = f"csv_files/{results_or_data}/{subreddit_name}_{scrape_data_type}_{results_or_data}.csv"

    results_df = use_csv_for_dataframe(filepath)

    create_histograms(results_df, subreddit_name)
    # create_violin_plot(results_df, subreddit_name)


# describe given csv

def csv_describe(subreddit_name, scrape_data_type):

    results_or_data = 'results'
    filepath = f"csv_files/{results_or_data}/{subreddit_name}_{scrape_data_type}_{results_or_data}.csv"
    
    df = use_csv_for_dataframe(filepath)
    print(df.describe())


# Packaging the code for comments #


# scrapes from scratch and visualise
# 100 posts and 100 comments = too many requests (10000)

def scrape_comments_and_plot_sentiment(subreddit_name, data_filter, time='year', number_of_posts=10, max_comments=10):

    global data_df, df

    # gather data (should be changed to depend on top, hot)
    data_df = subreddit_comments_scrape(subreddit_name, data_filter, time='year', number_of_posts=number_of_posts, max_comments=max_comments)
    # save the data (name: india_50_top_post_10_comments_data.csv)
    save_df_to_csv(data_df, filepath=f'csv_files\comment_data\{subreddit_name}_{number_of_posts}_{data_filter}_post_{max_comments}_comments_data.csv')

    # find results
    df = sentiment_results_all_comments(data_df)
    # save results (name: india_50_top_post_10_comments_result.csv)
    save_df_to_csv(df, 
                   filepath=f'csv_files\comment_result\{subreddit_name}_{number_of_posts}_{data_filter}_post_{max_comments}_comments_result.csv')

    # display a combined plot of violin, correlation and bar chart
    create_violin_plot(df,
                       title=f'r/{subreddit_name} {data_filter} post comments sentiment analysis', 
                       bgcolor='white',
                       filename=f'{subreddit_name}_{data_filter}_{number_of_posts}_comments_violin_chart')
    combined_plot(df, title=f'r/{subreddit_name} {data_filter} post comments sentiment analysis',
                  filename=f'{subreddit_name}_{data_filter}_{number_of_posts}_post_comments_combined_plots.png')


# just scrape (not analysing sentiment) and put the data in a folder

def scrape_comment_data_and_save(subreddit_name, data_filter, time='year', number_of_posts=10, max_comments=10):

    # gather data (should be changed to depend on top, hot)
    data_df = subreddit_comments_scrape(subreddit_name, data_filter, time=time, number_of_posts=number_of_posts, max_comments=max_comments)
    # save the data as csv
    save_df_to_csv(data_df, filepath=f'csv_files\comment_data\{subreddit_name}_{number_of_posts}_{data_filter}_post_comment_data.csv')


# use an already scraped data to find sentiment and visualise

def scraped_comment_data_visualise_sentiment(data_df_filepath, result_filename, title=''):
    
    # convert csv to data df
    data_df = use_csv_for_dataframe(data_df_filepath)

    # use df to get sentiment
    df = sentiment_results_all_comments(data_df)
    # save sentiment
    save_df_to_csv(df, 
                   filepath=f'csv_files\comment_result\{result_filename}')
    
    # visualise sentiment
    # create violin plot
    create_violin_plot(df, title=f"{title} sentiment violin plot")
    # create sentiment bar chart
    sentiment_bar_plot(df, title=f"{title} sentiment bar plot")
    # create correlation chart
    plot_correlation(df, title=f"{title} sentiment correlation plot")


# use already scraped result to visualise

def scraped_comment_result_visualise_sentiment(result_df_filepath, title=''):

    global df
    
    # convert csv to sentiment df
    df = use_csv_for_dataframe(result_df_filepath)

    # visualise sentiment
    # create violin plot
    create_violin_plot(df, title=f"{title} sentiment violin plot")
    # create bar chart
    sentiment_bar_plot(df, title=f"{title} sentiment bar plot")
    # create correlation chart
    plot_correlation(df, title=f"{title} sentiment correlation plot")