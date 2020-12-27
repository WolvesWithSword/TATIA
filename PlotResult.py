import math
import numpy as np
#graphique
import matplotlib.pyplot as plt

def autolabel(ax,rects):
    #Attach a text label above each bar in *rects*, displaying its height.
    for rect in rects: 
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=6, rotation=90)

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def plotTrueData(names, scores, support):

    scoresCut = [truncate(scr,2) for scr in scores]

    x = np.arange(len(names))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 10))
    rects1 = ax.bar(x - width/2, scoresCut, width, color='b', label='Precision')
    ax2 = ax.twinx()
    rects2 = ax2.bar(x + width/2, support, width, color='g', label='Support')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Precision (%)')
    ax2.set_ylabel('Number of support')
    ax.set_title('Precision of "True" for each categories')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    
    plt.legend(handles=[rects1, rects2], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    #label rotation
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    autolabel(ax,rects1)
    autolabel(ax2,rects2)

    fig.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.45, right=0.86, top=0.95, wspace=0.2, hspace=0.2)
    plt.savefig("true_data_plot.png")

    plt.show()

def plotData(names, scores, f1score):

    scoresCut = [truncate(scr,2) for scr in scores]
    f1Cut = [truncate(scr,2) for scr in f1score]

    x = np.arange(len(names))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 10))
    rects1 = ax.bar(x - width/2, scoresCut, width, color='b', label='precision')
    ax2 = ax.twinx()
    rects2 = ax2.bar(x + width/2, f1Cut, width, color='g', label='f1-score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Precision (%)')
    ax2.set_ylabel('F1-score (%)')
    ax.set_title('Precision and F1-score for each categories')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    
    plt.legend(handles=[rects1, rects2], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    #label rotation
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    autolabel(ax,rects1)
    autolabel(ax2,rects2)

    fig.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.45, right=0.86, top=0.95, wspace=0.2, hspace=0.2)

    plt.savefig("data_plot.png")

    plt.show()