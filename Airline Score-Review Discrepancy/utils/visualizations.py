import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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