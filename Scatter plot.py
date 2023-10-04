import pandas as pd
import matplotlib.pyplot as plt

# Read dataset into a DataFrame
df = pd.read_csv("E:\Foundation of Data Science\po2_data.csv")

columns = ['jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 
           'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 
           'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'nhr', 
           'hnr', 'rpde', 'dfa', 'ppe']

def plot_features(start, end, figure_number):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout()

    ax1.scatter(x = df[columns[start]], y = df['total_updrs'])
    ax1.set_xlabel(columns[start])
    ax1.set_ylabel('Total UPDRS Score')

    ax2.scatter(x = df[columns[start + 1]], y = df['total_updrs'])
    ax2.set_xlabel(columns[start + 1])
    ax2.set_ylabel('Total UPDRS Score')

    ax3.scatter(x = df[columns[start + 2]], y = df['total_updrs'])
    ax3.set_xlabel(columns[start + 2])
    ax3.set_ylabel('Total UPDRS Score')

    ax4.scatter(x = df[columns[start + 3]], y = df['total_updrs'])
    ax4.set_xlabel(columns[start + 3])
    ax4.set_ylabel('Total UPDRS Score')

    plt.show()

plot_features(0, 4, 1)
plot_features(4, 8, 2)
plot_features(8, 12, 3)
plot_features(12, 16, 4)
