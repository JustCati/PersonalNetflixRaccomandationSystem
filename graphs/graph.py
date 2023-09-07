import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def printTable(df, title):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.set_title(title, fontdict={"fontsize" : 20}, pad=20, loc="center", color="red", y=1.1)
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', rowLoc='center', colLoc='center')
    fig.tight_layout()
    plt.show()


def plotBestK():
    with open("bestK.txt", "r") as f:
        lines = f.readlines()

        counts = []
        bestKs = []
        for line in lines:
            count, bestK = line.split(", ")
            counts.append(int(count))
            bestKs.append(int(bestK))

        num = 0
        for k, elem in zip(bestKs, counts):
            if k == elem or k == elem - 1:
                num += 1
        perc = num / len(bestKs) * 100

        plt.plot([0, 50], [0, 50], color="red", linestyle="--")
        plt.plot(counts, bestKs, linestyle="solid", marker="o", color="blue")
        plt.xlabel("Numero di film nel Training Set")
        plt.ylabel("Migliore K")
        plt.grid(True)

        plt.xlim(1, 52)
        plt.ylim(1, 52)
        plt.xticks(np.arange(1, 52, 1))
        plt.yticks(np.arange(1, 52, 1))
        plt.tick_params(axis='both', which='major', labelsize=8)

        plt.text(5, 43, f"Percentuale di K corrispondente\nal massimo possibile (±1) per istanza: {perc:.0f}%", bbox=dict(facecolor='white', alpha=1))        
        plt.show()


def plotPerf():
    df = pd.read_csv("perf.csv")

    x = df["model"].to_list()
    y = df["rmse"].to_list()
    
    plt.bar(x[::2], y[::2], color="red")
    plt.bar(x[1::2], y[1::2], color="cyan")
    plt.xticks(rotation=45)
    plt.ylabel("RMSE")
    plt.legend(["Senza Bias", "Con Bias"], loc="lower right")
    plt.title("RMSE per ogni modello")

    for i, v in enumerate(y[::2]):
        plt.text(i - 0.2, v + 0.01, f"{v:.2f}", fontsize=8)
    for i, v in enumerate(y[1::2]):
        plt.text(i + 0.1, v + 0.01, f"{v:.2f}", fontsize=8)
    plt.show()



if __name__ == "__main__":
    # plotBestK()
    plotPerf()
    exit()
