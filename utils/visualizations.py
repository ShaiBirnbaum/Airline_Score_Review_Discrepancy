import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS


#returns bar histograms for 9 features 
def ratings_bar_hist(data, explain_vars):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, column in enumerate(explain_vars):
        sns.countplot(data=data, x=column, palette = ["#97e3fc"], ax=axes[i]) #hue='OverallScore'
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'Distribution of {column}')

    # Adjust the layout
    fig.tight_layout()

    # Show the plot
    plt.show()

#returns value counts of some features
    def vars_hist_table(data, rating_col):
        table = pd.DataFrame(columns=['rating', 'count'])
        for col in rating_col:
            value_counts = data[col].value_counts()
            table = table.append({'rating': col, 'count': value_counts.values[0]}, ignore_index=True)
        return table
    

#style data with review column
def reviews_style(data):
    pd.set_option('max_colwidth', 1000)
    pd.set_option("display.colheader_justify", "left")
    styled_data= data.style.set_properties(**{'text-align': 'left'})
    return styled_data

#scatters sample prediction against real scores and marks tagged samples.
def scatter_tag_deviations(data,x_col,y_col,hue):
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue, palette='viridis', s=100, alpha=0.2, linewidth=0.2)

    ## Add a 45-degree line for reference
    plt.plot([data[x_col].min(), data[x_col].max()], [data[x_col].min(), data[x_col].max()], 
             color='red', linestyle='--', alpha=0.4, label='45-degree line')

    # Add labels and title
    plt.xlabel('Actual Values (OverallScore)')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')

    # Show the legend
    plt.legend(title='Reviews', loc='lower right', labels=['Mismatch', 'Match'])

    # Show the plot
    plt.grid(True)
    plt.show()



#compare distribution of scores in general population vs in when feature is null
def compare_score_dist(data, score_col, condition_col):
    df_null_review = data[[score_col]][data[condition_col].isnull()]
    df_all_reviews = data[[score_col]]

    # Create a shared axis for both plots
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed

    # Plot both KDEs on the shared axis with bw_adjust
    sns.kdeplot(data=df_all_reviews[score_col], ax=ax, bw_method='scott', bw_adjust=2, label='All Reviews', color='blue')
    sns.kdeplot(data=df_null_review[score_col], ax=ax, label='Null Review', color='red')

    # Add labels and title
    ax.set_xlabel('Overall Score')
    ax.set_ylabel('Density')
    ax.set_title('KDE Plot of Overall Scores')

    # Set x-axis limits based on the minimum and maximum values in the 'OverallScore' column
    x_min = data[score_col].min()
    x_max = data[score_col].max()
    ax.set_xlim(x_min, x_max)

    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()

#boxen plot of many different ratings between 0 and 5
def ratings_boxen(data, boxen_cols):
    plt.figure(figsize=(10, 6))
    ax = sns.boxenplot(data=data[boxen_cols], palette='pastel')

    # Add annotations (for example, median values)
    medians = data[boxen_cols].median()
    for i, median in enumerate(medians):
        ax.text(i, median, f'Median: {median}', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Rating')
    plt.ylabel('Value')
    plt.title('Boxen Plot of Ratings')

    # Rotate x-axis labels to 45 degrees
    plt.xticks(rotation=45)

    plt.show()

#styled count plot of ratings column.
def styled_count_plot(data, column):
    # Create a copy of the DataFrame
    data_copy = data.copy()

    # Drop rows with missing or non-finite values in 'OverallScore'
    data_copy = data_copy.dropna(subset=[column])

    # Convert 'OverallScore' column to integers in the copied DataFrame
    data_copy[column] = data_copy[column].astype(int)

    # Create the value counts for 'OverallScore', sort by index in ascending order
    value_counts = data_copy[column].value_counts().sort_index(ascending=True)

    # Set the figure size
    plt.figure(figsize=(8, 4))

    # Create the bar plot
    ax = value_counts.plot.bar(color='skyblue', edgecolor='black', width=0.7)

    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Count of {column}', fontsize=16)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Apply the comma formatting to the y-axis ticks directly
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.show()

#takes text and filters punctuation ans stop words
def punctuation_stop(text):
    """Remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered

#filter unwanted words list and patterns
def filter_words(words_filtered, unwanted_words):
    # Filter out words containing unwanted_words
    words_filtered = [word for word in words_filtered if word not in unwanted_words]

    # Filter out consecutive words with the same pattern
    filtered_words = [words_filtered[0]]
    for i in range(1, len(words_filtered)):
        if words_filtered[i] != words_filtered[i - 1]:
            filtered_words.append(words_filtered[i])

    return filtered_words

#Generate a word cloud from a DataFrame column of text.
def generate_word_cloud(df, column, unwanted_words, max_words=200):
    """
    Parameters:
        df (DataFrame): The DataFrame containing the text column.
        column (str): The name of the text column in the DataFrame.
        unwanted_words (list): List of unwanted words to filter from the cloud.
        max_words (int, optional): The maximum number of words in the word cloud. Default is 200.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    data_copy = df.copy()

    # Remove rows with NaN values in the specified column temporarily
    data_copy = data_copy.dropna(subset=[column])

    # Convert the 'Title' column to strings and join
    words = " ".join(data_copy[column].astype(str))

    words_filtered = punctuation_stop(words)

    filtered_words = filter_words(words_filtered, unwanted_words)

    # Join the filtered words back into a single text string
    text = " ".join(filtered_words)

    wc = WordCloud(background_color="white", random_state=1, stopwords=STOPWORDS, max_words=max_words, width=1500, height=800)
    wc.generate(text)

    plt.figure(figsize=[10, 10])
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()

#styled Ticked Histogram
def styled_hist(col, bins_n):
    plt.hist(col, bins=bins_n, edgecolor='black')

    # Add labels and title
    plt.xlabel('Word Length')
    plt.ylabel('Number of Reviews')
    plt.title('Distribution of Word Lengths in Reviews')

    # Add grid lines
    plt.grid(True)

    # Format y-axis labels with commas for thousands
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Show the plot
    plt.tight_layout()
    plt.show()

#styled scatter
def styled_scatter(data, x, y):
    # Create a scatter plot
    data.plot.scatter(x, y, alpha=0.01)

    # Add labels and title
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    plt.title(f'{x} vs. {y}')

    # Set y-axis ticks to show all score values
    score_ticks = data[y].unique()
    plt.yticks(score_ticks)

    # Limit x-axis to 800
    plt.xlim(0, 800)

    # Show the plot
    plt.tight_layout()
    plt.show()