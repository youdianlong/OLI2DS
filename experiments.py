import time

from OLIDS import OLIDS
import matplotlib.pyplot as plt
import numpy as np
import preprocess


def check(data, dataset_name):
    if isinstance(data, list):
        label = np.array([item["class_label"] for item in data])
    else:
        label = np.array([item[-1] for item in data])
    unique_elements, counts_elements = np.unique(label, return_counts=True)
    label1 = counts_elements.tolist()[0]
    label2 = counts_elements.tolist()[1]
    print("Summary:\nname:{}, dataset:{}, label:{}, IR:{:.1f}".format(dataset_name, (len(data), len(data[0])),
                                            (unique_elements.tolist(), counts_elements.tolist()),
                                            min([label1, label2])/(label1 + label2)*100))


def plot_g_mean(plot_list, dataset):
    color_list = {"OLIDS": ["^", "red"]}
    for ele in plot_list:
        x = ele[1]
        y = ele[2]
        plt.plot(x[1:][0::int(len(x) / 30)], y[1:][0::int(len(x) / 30)], label=ele[0], marker=color_list[ele[0]][0],
                 linestyle="--",
                 color=color_list[ele[0]][1])
    plt.legend()
    plt.xlabel("Instance")
    plt.ylabel("G-mean")
    plt.title(dataset)
    figure_name = './figures/' + dataset + '_G_mean' + str("_") + time.strftime("%H%M%S") + '.png'
    plt.savefig(figure_name)
    plt.grid()
    plt.show()


def plot_f_measure(plot_list, dataset, style):
    color_list = {"OLIDS": ["^", "red"]}

    for ele in plot_list:
        x = ele[1]
        y = ele[2]
        px = x[1:][0::int(len(x) / 40)]
        py = y[1:][0::int(len(x) / 40)]
        plt.plot(px, py, label=ele[0], marker=color_list[ele[0]][0], linestyle="--",
                 color=color_list[ele[0]][1])
    plt.legend()
    plt.xlabel("Instance")
    plt.ylabel("F-measure")
    plt.title(dataset)
    figure_name = './figures/' + dataset + '_FMeasure' + str("_") + time.strftime("%H%M%S") + '.png'
    plt.savefig(figure_name)
    plt.grid()
    plt.show()


def experiment(data, dataset_name):
    check(data, dataset_name)
    modeDic = {
        1: "capricious",
        2: "trapezoidal",
    }
    GMeanList, FMeasureList, AccuracyList, LossList = [], [], [], []
    mode = 1
    sparse = 0
    Lambda = 30
    C = 0.0100000
    B = 1
    theta = 8
    gama = 0

    Result = OLIDS(data, C, Lambda, B, theta, gama, sparse, modeDic[mode]).fit()

    x = list(range(len(Result[0])))
    
    GMeanList.append(("OLIDS", x, Result[0]))
    FMeasureList.append(("OLIDS", x, Result[1]))

    plot_g_mean(GMeanList, dataset_name)
    plot_f_measure(FMeasureList, dataset_name, modeDic[mode])
    

def streamOverUCIDatasets():
    experiment(preprocess.readWdbcNormalized(), "WDBC")
    # experiment(preprocess.readCreditA(), "Credit-a")
    # experiment(preprocess.readSvmguide3Normalized(), "svmguide3")
    # experiment(preprocess.readKrVsKp(), "kr-vs-kp")
    # experiment(preprocess.readSplice(), "splice")
    # experiment(preprocess.readSpambaseNormalized(), "Spambase")


start_time = time.time()
streamOverUCIDatasets()
print("--- %s seconds ---" % (time.time() - start_time))
