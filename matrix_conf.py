import numpy as np
import os
import argparse
import multiprocessing
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import time
import random
import seaborn as sn
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from math import ceil


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("file_npy",
                        help="Folder with npys files", nargs="+")
    parser.add_argument("file",
                        help="File with true labels")
    parser.add_argument("file_class",
                        help="File with classes")
    parser.add_argument("-xl", "--extra_labels",
                        help="File with extra true labels for other streams",
                        nargs="+")
    parser.add_argument("-t", "--threads",
                        help="Number of threads", default=-1, type=int)
    parser.add_argument("-n", "--name",
                        help="Basename to save figures", default="chart")
    parser.add_argument("-sn", "--streams_names",
                        help="Labels for legend (normally the names of \
                        each stream)", default=["Accuracy"],
                        nargs="+")
    parser.add_argument("-ms", "--multi_stream",
                        help="Number of the streams in the multi-stream arch",
                        type=int, default=1)
    parser.add_argument("-w", "--weights",
                        help="Weights of each stream in a multi stream method",
                        nargs="+", type=float)
#    parser.add_argument("-s", "--similarity",
#                        help="Calculates similarity between classifiers 2x2",
#                        action="store_true")
    parser.add_argument("-m", "--mode",
                        help="Chart mode:\ndiff: create the differences chart \
                        \nconf: create the confusion matrix chart \nall: \
                        create all charts of the previous options",
                        choices=["diff", "conf", "all"])
    parser.add_argument("-o", "--out_dir",
                        help="Output directory", default="charts_output")
    return parser.parse_args()


def np_softmax(X):

    return np.exp(X)/np.sum(np.exp(X))


def plot_matrix(cm, class_names, out_dir, fig_size=(10, 7),
                font_size='x-small', cmap=None, name="chart"):

    df_cm = pd.DataFrame(cm, index=np.arange(len(class_names))+1,
                         columns=np.arange(len(class_names))+1)
    fig = plt.figure(figsize=fig_size)

    if(cmap is None):
        hm = sn.heatmap(df_cm, vmin=0.0, vmax=1.0, annot=False)
        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0,
                                ha='right', fontsize=font_size)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0,
                                ha='center', fontsize=font_size)
    else:
        hm = sn.heatmap(df_cm, annot=False, vmin=0.0, vmax=1.0, cmap=cmap)
        hm.yaxis.set_ticklabels(hn.yaxis.get_ticklabels(), rotation=0,
                                ha='right', fontsize=font_size)
        hm.xaxis.set_ticklabels(hn.xaxis.get_ticklabels(), rotation=45,
                                ha='right', fontsize=font_size)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig.savefig(os.path.join(out_dir, name+"_conf_mat.pdf"),
                bbox_inches='tight')

    return fig


def plot_bar(cm, class_names, out_dir, fig_size=(10, 7), font_size=14,
             cmap=None, name="chart", labels=None, cm_list=None):

    # extract the size to make the ticks
    n_groups = len(class_names)
    # extract the main diagonal of the confusion matrix
    main_diag = np.diagonal(cm)

    fig, ax = plt.subplots()

    # generate de range of the bars
    index = np.arange(n_groups)
    # generate spaced ticks
    ticks = np.linspace(0, n_groups - 1, n_groups//10 + 1, dtype=np.int16)
    # add the first tick
    #ticks = np.insert(ticks, 0, 1, axis=0)
    bar_width = 0.8

    opacity = 0.4

    # split the negative from the positive values of the main diagonal
    idx_pos = [idx for idx, value in enumerate(main_diag) if value >= 0]
    diag_pos = [value for idx, value in enumerate(main_diag) if value >= 0]
    idx_neg = [idx for idx, value in enumerate(main_diag) if value < 0]
    diag_neg = [value * (-1) for idx, value in enumerate(main_diag)
                if value < 0]

    # if there is negative values this is the bar plot of differences
    if len(idx_neg) == 0:

        # default bar plot
        if cm_list is None:
            rects1 = ax.bar(index, main_diag, bar_width, edgecolor="black",
                            linewidth=0, alpha=opacity,
                            color=(153/255., 153/255., 1), label="Accuracy")
        # bar plot of individual contributions
        else:
            # temp_color = 153/255.
            temp_color = 0.
            colors = [(temp_color, temp_color, 1.),
                      (1., temp_color, temp_color),
                      (temp_color, 1., temp_color), (temp_color, 1., 1.),
                      (1., 1., temp_color), (1., temp_color, 1.)]
            val1 = 92./255.
            val2 = 153./255.
            val3 = 194./255.
            comb_colors = [(val2, val2, 1.0), (1.0, val2, val2),
                           (val2, 1.0, val2), (val3, val1, val2),
                           (val1, val3, val2), (val2, val3, val1)]
            # maximuns to know who contributed more in each class
            '''maxes = [-1 for _ in range(n_groups)]
            idxs = [-1 for _ in range(n_groups)]
            for j, item in enumerate(cm_list):
                diag = np.diagonal(item)
                for i in range(n_groups):
                    if diag[i] > maxes[i]:
                        maxes[i] = diag[i]
                        idxs[i] = j'''

            if len(cm_list) == 9:
                patch = [None, None, None, None, None, None, None]
                cont = 0

                for i in range(3):
                    diag = (np.diagonal(cm_list[i]) + np.diagonal(cm_list[i+3])
                            + np.diagonal(cm_list[i + 6])) / 3.0
                    rects = ax.bar(idx_pos, diag, bar_width, alpha=opacity,
                                   color=colors[i], label=labels[i])
                    patch[cont] = mpatches.Patch(color=comb_colors[cont],
                                                 label=labels[i])
                    cont += 1

                for i in range(3):
                    for j in range(i+1, 3):
                        patch[cont] = mpatches.Patch(color=comb_colors[cont],
                                                     label='{} + {}'.format(
                                                     labels[i], labels[j]))

                        cont += 1
                patch[-1] = mpatches.Patch(color=(116./255., 157./255.,
                                           92./255.), label='{} + {} + {}\
                                           '.format(labels[0], labels[1],
                                           labels[2]))
                plt.legend(handles=patch, framealpha=1, loc='lower center',
                           ncol=7)

            else:
                for j in range(len(cm_list)):
                    # idx_pos = [idx for idx, value in enumerate(idxs)
                    #           if value == j]
                    # diag_pos = [main_diag[idx] for idx, value in
                    #            enumerate(idxs) if value == j]
                    # print("{} won in {} classes".format(labels[j],
                    #       len(diag_pos)))
                    diag = np.diagonal(cm_list[j])
                    rects = ax.bar(idx_pos, diag, bar_width, alpha=opacity,
                                   color=colors[j], label=labels[j])

    else:
        opacity = 1.0
        rects1 = ax.bar(idx_pos, diag_pos, bar_width, alpha=opacity,
                        color=(153/255., 153/255., 1), label=labels[0])

        print("{} won in {} classes".format(labels[0],
              len(diag_pos) - np.count_nonzero(main_diag == 0)))

        rects2 = ax.bar(idx_neg, diag_neg, bar_width, alpha=opacity,
                        color=(1., 153/255., 153/255.), label=labels[1])

        print("{} won in {} classes".format(labels[1], len(diag_neg)))
        print("Tie in {} classes".format(np.count_nonzero(main_diag == 0)))

    # [rects1[idx].set_color((1., 153/255., 153/255.)) for idx, value in
    #    enumerate(main_diag) if value < 0]

    # automagically set the max accuracy
    min_y, max_y = 0.0, (ceil(10*max(np.abs(main_diag))))/10.0
    print(np.abs(main_diag))
    print(max(np.abs(main_diag)))
    print("Max_y: {}".format(max_y))
    ax.set_xlabel('Classes')
    ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy by class')
    # ticks are 0-indexed
    ax.set_xticks(ticks)
    ax.set_ylim(min_y, max_y)
    # but tickslabels doesnt
    ax.set_xticklabels(ticks+1, rotation=0)

    '''min_diag = np.min(main_diag)
    max_diag = np.max(main_diag)

    if min_diag >= 0 :
        y_ticks = np.linspace(0., 1., 11)
    else:
        y_ticks = np.linspace(-1., 1., 11)'''
    y_ticks = np.linspace(min_y, max_y, (max_y - min_y)*100//(10) + 1)

    ax.set_yticks(y_ticks)
    # make the bars fill all available space in both dimensions
    plt.margins(0.0, 0.0)
    # make horizontal lines aligned to horizontal ticks
    plt.hlines(y_ticks, 0, n_groups, linestyles="dashed", linewidth=1)
    ax.legend(facecolor=(1.0, 1.0, 1.0))

    # fig.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    fig.savefig(os.path.join(out_dir, name+"_acc_bar.pdf"),
                bbox_inches='tight')


def calc_acc(Y, y_pred):
    arr = list(map(float, np.equal(Y, y_pred)))
    test_acc = sum(arr)/len(arr)
    return test_acc


def np_softmax(X):
    return np.exp(X)/np.sum(np.exp(X))


def load_label(filename, classes=None):

    assert (os.path.exists(filename))
    labels = []
    classes = list(classes)

    if filename.endswith(".csv"):
        f = pd.read_csv(filename, header=None).values

        for line in f:
            labels.append(line[1])

    elif filename.endswith(".txt"):
        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines:
            idx = classes.index(line.split("/")[0])
            labels.append(idx)

    return labels


def make_confusion_matrix(Y, Y_pred, classes, out_dir, plot=False, normalize=True,
                          name="chart"):

    cm = confusion_matrix(Y, Y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = calc_acc(Y, Y_pred)

    print("Test accuracy: {:5.2f}%".format(accuracy*100))

    if(plot):
        fig = plot_matrix(cm, classes, out_dir, name=name)

    else:
        target_names = [str(i) for i in list(classes)]
        print(classification_report(Y, Y_pred, target_names=target_names))

    return cm, accuracy


def similarity(preds, labels, ground_truth):

    assert len(preds) == len(labels)
    assert len(preds) >= 2
    for i in range(len(preds)):
        for j in range(i+1, len(preds)):
            cf = []
            try:
                assert len(preds[i]) == len(preds[j])
            except AssertionError:
                print("({}) {} != {} ({})".format(labels[i], len(preds[i]),
                      len(preds[j]), labels[j]))
            n_samples = len(preds[i])

            sim = np.equal(preds[i], preds[j])
            tot_sim = np.sum(sim)/(n_samples)
            print("Total Similarity between {} and {}: {:05.2f}%".format(
                    labels[i], labels[j], tot_sim*100))

            acc1 = np.equal(ground_truth, preds[i])
            acc2 = np.equal(ground_truth, preds[j])

            acc_sim = np.sum(np.logical_and(sim, acc1))/(n_samples)

            assert np.sum(np.equal(np.logical_and(sim, acc1),
                                   np.logical_and(sim, acc2))) == len(sim)
            print("Accuracy Similarity between {} and {}: {:05.2f}%".format(
                    labels[i], labels[j], acc_sim*100))

            arr1 = list(map(float, acc1))
            arr2 = list(map(float, acc2))

            cf.append(arr1)
            cf.append(arr2)
            cf_np = np.any(np.array(cf), axis=0).astype(int)
            print("Maximun possible acc: {:05.2f}%".format((
                    np.sum(cf_np)/cf_np.shape[0])*100))


def conf_matrix(args, n, classes, y, out_dir):

    if n == 1:  # simple confusion matrix

        npy = args.file_npy[0]
        X = np.load(npy)
        print("X shape: {}".format(X.shape))
        y_pred = np.argmax(X, axis=1)

        cm, accuracy = make_confusion_matrix(y, y_pred, classes, out_dir,
                                             plot=True, name=args.name)

    else:
        if n == args.multi_stream and args.multi_stream == len(args.weights):
            # multi-stream confusion matrix
            X_mean = []
            cm_mean = []
            n_streams = args.multi_stream

            X_mean = []
            # seleciono as streams de cada split e o respectivos pesos
            npys = args.file_npy
            weights = args.weights

            # percorro os npys e pesos das streams paralelamente
            for f, w in zip(npys, weights):

                # carrego o npy
                X = np.load(f)
                print("X shape: {}".format(X.shape))

                # faco a ponderacao dos npys
                if len(X_mean) == 0:
                    X_mean = X * w
                else:
                    X_mean += X * w

                # verifico qual a predicao da ponderacao dentro de cada split
                y_pred_mean = np.argmax(X_mean, axis=1)
                cm, accuracy = make_confusion_matrix(y, y_pred_mean, out_dir,
                                                     classes, plot=False,
                                                     name=args.name,
                                                     normalize=False)
                print("Total samples: {}".format(np.sum(cm)))

                # acumulo as matrizes de confusao de cada split
                if len(cm_mean) == 0:
                    cm_mean = cm
                else:
                    cm_mean += cm

            print("Total samples: {}".format(np.sum(cm_mean)))
            print("Got right: {}".format(np.sum(np.diag(cm_mean))))
            print("Acc: {}".format(np.sum(np.diag(cm_mean))/np.sum(cm_mean)))
            cm_mean = cm_mean.astype('float') / np.sum(cm_mean, axis=1)
            main_diag = np.diagonal(cm_mean)

            print("Final mean accuracy: {:05.4f}".format(np.mean(main_diag)))
            plot_matrix(cm_mean, classes, out_dir, name="{}_mean".format(args.name))

        elif n == 3 and len(args.weights) == 1 and len(args.extra_labels) == 2:
            # all splits of one stream confusion matrix
            X_mean = []
            cm_mean = []
            n_streams = args.multi_stream

            for i in range(3):
                X_mean = []
                # seleciono o npy de cada split
                npys = args.file_npy[i]

                if i > 0:
                    y = load_label(args.extra_labels[i-1], classes)

                X = np.load(npys)
                print("X shape: {}".format(X.shape))

                # verifico qual a predicao
                y_pred = np.argmax(X, axis=1)
                cm, accuracy = make_confusion_matrix(y, y_pred, classes,
                                                     out_dir, plot=False,
                                                     name=args.name,
                                                     normalize=False)
                print("Total samples: {}".format(np.sum(cm)))

                # acumulo as matrizes de confusao
                if len(cm_mean) == 0:
                    cm_mean = cm
                else:
                    cm_mean += cm

            print("Total samples: {}".format(np.sum(cm_mean)))
            print("Got right: {}".format(np.sum(np.diag(cm_mean))))
            print("Acc: {}".format(np.sum(np.diag(cm_mean))/np.sum(cm_mean)))
            cm_mean = cm_mean.astype('float') / np.sum(cm_mean, axis=1)
            main_diag = np.diagonal(cm_mean)

            print("Final mean accuracy: {:05.4f}".format(np.mean(main_diag)))
            plot_matrix(cm_mean, classes, out_dir, name="{}_mean".format(args.name))

        elif (n == 3 * args.multi_stream and
              args.multi_stream == len(args.weights) and
              len(args.extra_labels) == 2):
            # all splits multi-stream confusion matrix

            X_mean = []
            cm_mean = []
            n_streams = args.multi_stream

            for i in range(3):
                X_mean = []
                # seleciono as streams de cada split e o respectivos pesos
                npys = args.file_npy[i*n_streams:(i+1)*n_streams]
                weights = args.weights

                if i > 0:
                    y = load_label(args.extra_labels[i-1], classes)

                # percorro os npys e pesos das streams paralelamente
                for f, w in zip(npys, weights):

                    # carrego o npy
                    X = np.load(f)
                    print("X shape: {}".format(X.shape))

                    # faco a ponderacao dos npys
                    if len(X_mean) == 0:
                        X_mean = X * w
                    else:
                        X_mean += X * w

                # verifico qual a predicao da ponderacao dentro de cada split
                y_pred_mean = np.argmax(X_mean, axis=1)
                cm, accuracy = make_confusion_matrix(y, y_pred_mean, classes,
                                                     out_dir, plot=False,
                                                     name=args.name,
                                                     normalize=False)
                print("Total samples: {}".format(np.sum(cm)))

                # acumulo as matrizes de confusao de cada split
                if len(cm_mean) == 0:
                    cm_mean = cm
                else:
                    cm_mean += cm

            print("Total samples: {}".format(np.sum(cm_mean)))
            print("Got right: {}".format(np.sum(np.diag(cm_mean))))
            print("Acc: {}".format(np.sum(np.diag(cm_mean))/np.sum(cm_mean)))
            cm_mean = cm_mean.astype('float') / np.sum(cm_mean, axis=1)
            main_diag = np.diagonal(cm_mean)

            print("Final mean accuracy: {:05.4f}".format(np.mean(main_diag)))
            plot_matrix(cm_mean, classes, out_dir, name="{}_mean".format(args.name))


def diff_chart(args, n, classes, y, out_dir):

    if n == 2:  # simple difference chart

        cms = []
        arr = []
        cf = []

        for i in range(n):
            file_npy = args.file_npy[i]
            X = np.load(file_npy)
            print("X shape: {}".format(X.shape))
            y_pred = np.argmax(X, axis=1)

            cm, accuracy = make_confusion_matrix(y, y_pred, classes, out_dir,
                                                 plot=False, name=args.name
                                                 + str(i), normalize=True)
            # print(np.diagonal(cm))
            cms.append(cm)

            arr = list(map(float, np.equal(y, y_pred)))
            cf.append(arr)

        # cf_np = np.any(np.array(cf), axis=0).astype(int)
        # print("Maximun possible acc: {}".format(
        #        np.sum(cf_np)/cf_np.shape[0]))

        diff = cms[0] - cms[1]
        plot_bar(diff, classes, out_dir, name=args.name + "_diff",
                 labels=args.streams_names)

    elif n == 6:  # difference chart across splits

        cms = []
        arr = []
        cf = []
                   
        for i in range(2):

            cm_mean = []

            for j in range(3):
                
                file_npy = args.file_npy[j*2+i]
                X = np.load(file_npy)
                print("X shape: {}".format(X.shape))
                y_pred = np.argmax(X, axis=1)
    
                cm, accuracy = make_confusion_matrix(y, y_pred, classes,
                                                     out_dir, plot=False,
                                                     name=args.name + str(i))
                
                # acumulo as matrizes de confusao de cada split
                if len(cm_mean) == 0:
                    cm_mean = cm
                else:
                    cm_mean += cm
            
            cm = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis]
            cms.append(cm)

            arr = list(map(float, np.equal(y, y_pred)))
            cf.append(arr)

        # cf_np = np.any(np.array(cf), axis=0).astype(int)
        # print("Maximun possible acc: {}".format(
        #        np.sum(cf_np)/cf_np.shape[0]))

        diff = cms[0] - cms[1]
        plot_bar(diff, classes, out_dir, name=args.name + "_diff",
                 labels=args.streams_names)


def main(args):

    dir_to_save_imgs = args.out_dir
    os.makedirs(dir_to_save_imgs, exist_ok=True)

    num_files = len(args.file_npy)

    classes = np.loadtxt(args.file_class, dtype=str, usecols=[1])
    y = load_label(args.file, classes)

    # check the mode
    if args.mode == 'all':
        conf_matrix(args, num_files, classes, y, dir_to_save_imgs)
        diff_chart(args, num_files, classes, y, dir_to_save_imgs)

    elif args.mode == 'diff':
        diff_chart(args, num_files, classes, y, dir_to_save_imgs)

    elif args.mode == 'conf':
        conf_matrix(args, num_files, classes, y, dir_to_save_imgs)


if __name__ == '__main__':

    args = get_args()
    main(args)
