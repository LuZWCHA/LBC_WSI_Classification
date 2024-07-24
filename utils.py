import math
import shutil
import logging
import numpy as np
from scipy import interp

import matplotlib
from sklearn.metrics import roc_curve, auc, classification_report

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ConfusionMatrix(object):
    def __init__(self, classes):
        self.confusion_matrix = torch.zeros(len(classes), len(classes))
        self.confusion_matrix_binary = torch.zeros(2, 2)
        self.classes = classes

    def update_matrix(self, preds, targets):
        preds = np.argmax(preds, axis=-1)
        # print(preds)
        # preds = torch.max(preds, 1)[1].cpu().numpy()
        # targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.confusion_matrix[t, p] += 1

            p = 1 if p > 0 else 0
            t = 1 if t > 0 else 0
            self.confusion_matrix_binary[t, p] += 1

    def plot_confusion_matrix(self, normalize=True, save_path='./Confusion Matrix.jpg'):
        cm = self.confusion_matrix.numpy()
        classes = self.classes
        num_classes = len(classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        im = plt.matshow(cm, cmap=plt.cm.Blues)  # cm.icefire
        plt.xticks(range(num_classes), classes, fontproperties="Times New Roman", fontsize=14)
        plt.yticks(range(num_classes), classes, fontproperties="Times New Roman", fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        for i in range(len(classes)):
            tempSum = 0
            for j in range(num_classes - 1):
                tempS = cm[i, j]
                tempSum += tempS
                color = 'white' if tempS > 50 else 'black'
                if cm[i, j] != 0:
                    plt.text(j, i, format(tempS, '0.2f'), color=color, ha='center', va="center", fontsize=15)
            tempS = 100 - tempSum
            color = 'white' if tempS > 50 else 'black'
            if float(format(abs(tempS), '0.2f')) != 0:
                plt.text(num_classes - 1, i, format(abs(tempS), '0.2f'), color=color, ha='center', va="center",
                         fontsize=15)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=12)
        cb.set_ticks(np.linspace(0, 100, 6))
        cb.set_ticklabels(('0', '20', '40', '60', '80', '100'))

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_confusion_matrix_binary(self, normalize=True, save_path='./Confusion Matrix.jpg'):
        cm = self.confusion_matrix_binary.numpy()
        classes = ['neg', 'pos']
        num_classes = 2
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        im = plt.matshow(cm, cmap=plt.cm.Blues)  # cm.icefire
        plt.xticks(range(num_classes), classes, fontproperties="Times New Roman", fontsize=14)
        plt.yticks(range(num_classes), classes, fontproperties="Times New Roman", fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        for i in range(len(classes)):
            tempSum = 0
            for j in range(num_classes - 1):
                tempS = cm[i, j]
                tempSum += tempS
                color = 'white' if tempS > 50 else 'black'
                if cm[i, j] != 0:
                    plt.text(j, i, format(tempS, '0.2f'), color=color, ha='center', va="center", fontsize=15)
            tempS = 100 - tempSum
            color = 'white' if tempS > 50 else 'black'
            if float(format(abs(tempS), '0.2f')) != 0:
                plt.text(num_classes - 1, i, format(abs(tempS), '0.2f'), color=color, ha='center', va="center",
                         fontsize=15)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=12)
        cb.set_ticks(np.linspace(0, 100, 6))
        cb.set_ticklabels(('0', '20', '40', '60', '80', '100'))

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()


class AUCMetric(object):
    def __init__(self, classes):
        self.targets = []
        self.preds = []
        self.classes = np.arange(len(classes))
        self.classes_list = classes

    def update(self, preds, targets):
        # preds = torch.softmax(preds.cpu(), dim=-1).detach().numpy()
        # targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.preds.append(p)
            self.targets.append(t)

    def calc_auc_score(self):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)
        micro_auc = metrics.roc_auc_score(targets, preds, average='micro')
        macro_auc = metrics.roc_auc_score(targets, preds, average='macro')
        return micro_auc, macro_auc

    def plot_roc_curve(self, save_path):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(self.classes)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(np.array(targets)[:, i], preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"], roc_auc["macro"] = self.calc_auc_score()
        # Plot all ROC curves
        plt.figure()

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.3f})'.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = ['aqua', 'darkorange', 'cornflowerblue', 'palegreen', 'lightcoral']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.3f})'.format(self.classes_list[i], roc_auc[i]))
        #
        # plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label='AUC={:.3f}'.format(auc))
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right')
        plt.savefig(save_path)
        plt.clf()
        plt.close()


class ClassificationMetrics(object):
    def __init__(self, classes):
        self.targets = []
        self.preds = []
        self.classes = classes

    def update(self, preds, targets):
        preds = np.argmax(preds, axis=-1)
        # preds = torch.argmax(preds.cpu(), dim=-1).detach().numpy()
        # targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.preds.append(p)
            self.targets.append(t)

    def report_metrics(self):
        print(classification_report(self.targets, self.preds, digits=4))
        
    
#! pip install imagehash
import imagehash
import cv2
def image_hash(img_path: str):
    img = cv2.imread(img_path)
    img_hash = imagehash.average_hash(img)
    return img_hash