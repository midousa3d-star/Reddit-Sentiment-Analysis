import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="id",
    client_secret="secret",
    user_agent="name",
)



# Posts scraping

#scrape top posts
def subreddit_top_post_scrape_df(subreddit_name, time='year', number_of_posts=100): 
    pd.set_option('max_colwidth', None)
    df = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.top(time, limit=number_of_posts):
        df.append([post.title, post.id, post.score])

    df = pd.DataFrame(df, columns=['text', 'post id', 'score'])
    return df

#scrape hot posts
def subreddit_hot_post_scrape_df(subreddit_name, number_of_posts=100): 
    pd.set_option('max_colwidth', None)
    df = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=number_of_posts):
        df.append([post.title, post.id, post.score])

    df = pd.DataFrame(df, columns=['text', 'post id', 'score'])
    return df


# combining both to a single function
def subreddit_posts_scrape(subreddit_name, data_filter='top', time='year', number_of_posts=100):
    if data_filter == 'top':
        df = subreddit_top_post_scrape_df(subreddit_name, time, number_of_posts)
    elif data_filter == 'hot':
        df = subreddit_hot_post_scrape_df(subreddit_name, number_of_posts)

    return df


# Comments scraping

# creates dataframe with columns as comment, score, post id
max_comments_amount=20
def subreddit_comments_scrape(subreddit_name, data_filter='top', time='year', number_of_posts=2, max_comments=max_comments_amount):

    # pd.set_option('max_colwidth', None)

    subreddit = reddit.subreddit(subreddit_name)
    comments_list = []

    # scrape comments, post id and score
    if data_filter == 'top':
        for post in subreddit.top(time, limit=number_of_posts):
            post.comments.replace_more(limit=100)

            for comment in post.comments.list()[:max_comments]:
            # for comment in post.comments.list():
                comment_info = {
                    'text': comment.body,
                    'score': comment.score,
                    'post id': post.id
                }
                comments_list.append(comment_info)

    elif data_filter == 'hot':
        for post in subreddit.hot(limit=number_of_posts):
            post.comments.replace_more(limit=100)

            for comment in post.comments.list()[:max_comments]:
            # for comment in post.comments.list():
                comment_info = {
                    'text': comment.body,
                    'score': comment.score,
                    'post id': post.id
                }
                comments_list.append(comment_info)
    
    # Create a DataFrame from the list of dictionaries
    comment_data_df = pd.DataFrame(comments_list)

    return comment_data_df


