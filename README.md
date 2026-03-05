# Reddit Sentiment Analysis
This application scrapes data from Reddit, performs sentiment analysis on the scraped posts or comments, and visualizes the results. It plots sentiment distributions, generates word clouds, and identifies clusters of texts with similar patterns.

## Features
- Scrape Reddit post titles or comments from specified subreddits.
- Perform sentiment analysis on the scraped data.
- Visualize sentiment distribution with violin plots and bar charts.
- Generate word clouds from the text data.
- Cluster sentences and display examples from each cluster.
- User-friendly interface for selecting subreddit, data type, and filters.

## Prerequisites
- Python 3.7 or higher
- Virtual environment named `.venv`
- Required Python packages (listed in `requirements.txt`)

## Setup

### Clone the Repository
```bash
git clone https://github.com/Sal-Faris/Reddit_Sentiment_Analysis.git
cd reddit-sentiment-analysis
```

### Create and Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Obtain client id, client secret and agent name
go to https://old.reddit.com/prefs/apps and create an app to get the client id, client secret and agent name and fill in the corresponding variables in ```reddit_scraping.py```

## Usage

### Run the Application
```bash
streamlit run app/gui.py
```
### Application Interface
- **Subreddit name**: Enter the name of the subreddit you want to analyze (e.g., `news`, `aww`, `datascience`).
- **Data to scrape**: Select whether to scrape post titles or comments.
- **Post type**: Choose the type of posts to scrape (Top or Hot).
- **Filter by time (if applicable)**: Select the time filter for Top posts (e.g., Hour, Day, Week, Month, Year, All).
- **Maximum number of posts**: Use the slider to select the maximum number of posts to scrape (1 to 1000).
- **Analyse button**: Click to start scraping, analyzing, and visualizing the data.

### Example Output

![](https://github.com/Sal-Faris/Reddit_Sentiment_Analysis/blob/main/SentimentDEMO.gif)
