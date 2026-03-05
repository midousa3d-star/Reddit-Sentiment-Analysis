def generate_word_cloud(filename):

    # importing all necessary modules
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    import pandas as pd


    df = pd.read_csv(filename, encoding ="latin-1")
    
    comment_words = ''
    stopwords = set(STOPWORDS)
    
    # iterate through the csv file
    for val in df.text:
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words += " ".join(tokens)+" "

    
    wordcloud = WordCloud(width = 1280, height = 720,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)
    
    # plot the WordCloud image                       
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    plt.show()



def generate_word_cloud_df(df):

    # importing all necessary modules
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    import pandas as pd

    
    comment_words = ''
    stopwords = set(STOPWORDS)
    
    # iterate through the csv file
    for val in df.text:
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words += " ".join(tokens)+" "

    
    wordcloud = WordCloud(width = 1280, height = 720,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)
    
    return wordcloud
    
    # # plot the WordCloud image                       
    # plt.figure(figsize = (8, 8), facecolor = None)
    # plt.imshow(wordcloud)
    # plt.axis("off")
    # plt.tight_layout(pad = 0)
    
    # plt.show()
