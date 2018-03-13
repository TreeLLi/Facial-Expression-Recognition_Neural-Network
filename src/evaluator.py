import matplotlib.pyplot as plt
import numpy as np
import os
# plot

def draw_loss_acc(loss_history, train_acc_history, val_acc_history, name=None):
    fig = plt.figure()
    sub1 = fig.add_subplot(211)
    sub1.set_title("Training loss")
    sub1.plot(loss_history,"o")
    sub1.set_xlabel('Iteration')

    sub2 = fig.add_subplot(212)
    sub2.set_title('Accuracy')
    sub2.plot(train_acc_history, '-o', label = 'train')
    sub2.plot(val_acc_history,'-o',label='val')
    sub2.plot([0.5]*len(val_acc_history),'k--')
    sub2.set_xlabel('Epoch')
    sub2.legend(loc='lower right')
    fig.set_size_inches(15,12)
    
    path = "figures/"
    if name is not None:
        splited = name.split('/')
        if len(splited) > 1:
            sub_path = splited[0] + '/'
            name = splited[1]
            path += sub_path
    if not os.path.exists(path):
        os.makedirs(path)
        
    # name = name+"_loss_acc.jpg" if name is not None else "loss_acc.jpg"
    name = name+"_loss_acc.jpg" if name is not None else "loss_acc.jpg"
    fig.savefig(path + name)
    # fig.show()
    
# Evaluation
    
def confusion_matrix (pre_class,act_class,label_num):
    confusion = np.zeros((label_num,label_num))
    # print(confusion)
    for index in range(len((pre_class))):
        actual_i = act_class[index]
        predict_j = pre_class[index]
        confusion[actual_i][predict_j] += 1
    return confusion

def cr(confusion_matrix,label_num):
    dig_mat = np.diag(np.ones(label_num))
    correct_class_num = sum(map(sum, dig_mat * confusion_matrix))
    total_num = sum(map(sum, confusion_matrix))
    return correct_class_num / total_num

def sub_cf(cf, index):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(cf.shape[0]):
        for j in range(cf.shape[0]):
            amount = cf[i][j]
            tp += amount if i==index and j==index else 0
            tn += amount if i!=index and j!=index else 0
            fp += amount if i!=index and j==index else 0
            fn += amount if i==index and j!=index else 0
    return np.asarray([[tp, fn], [fp, tn]])

def recall(sub_cf):
    tp = sub_cf[0][0]
    fn = sub_cf[0][1]
    if tp+fn == 0:
        return 0
    else:
        return tp / (tp + fn)

def precision(sub_cf):
    tp = sub_cf[0][0]
    fp = sub_cf[1][0]
    if tp+fp == 0:
        return 0
    else:
        return tp / (tp + fp)

def fas(recalls, precisions):
    return 2*recalls*precisions / (recalls+precisions)
    
def rates(cf):
    recalls = []
    precisions = []
    crs = []
    for i in range(cf.shape[0]):
        subcf = sub_cf(cf, i)
        recalls.append(recall(subcf))
        precisions.append(precision(subcf))
        crs.append(cr(subcf, 2))
    recalls = np.asarray(recalls)
    precisions = np.asarray (precisions)
    f1s = fas(recalls, precisions)
    return recalls, precisions, f1s, np.asarray(crs)


def evaluate(pre_class, act_class, label_num, name=None):
    cf = confusion_matrix(pre_class, act_class, label_num)

    recalls, precisions, f1s, crs = rates(cf)
    overall_cr = cr(cf, label_num)

    path = "evaluations/"
    name = name+"_eva.txt" if name is not None else "eva.txt"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+name, 'w') as f:
        write_cf(f, cf)
        write_rates(f, recalls, precisions, f1s, crs)

        f.write("\nThe overall classification rate:\n")
        f.write("{:.4f}\n".format(overall_cr))
        
    return cf, recalls, precisions, f1s, crs, overall_cr

def write_cf(f, cf):
    f.write("The confusion matrix:\n")
    num = cf.shape[0]
    
    for i in range(num):
        line = ""
        for j in range(num):
            line += "{:5} ".format(int(cf[i][j]))
        line += "{:5}\n".format(int(np.sum(cf[i])))
        f.write(line)

    total = np.sum(cf, axis=0)
    line = ""
    for i in range(num):
        line += "{:5} ".format(int(total[i]))
    line += "\n"
    f.write(line)
    
def write_rates(f, recalls, precisions, f1s, crs):
    num = recalls.shape[0]
    
    rec_line = ""
    pre_line = ""
    f1s_line = ""
    crs_line = ""
    for i in range(num):
        rec_line += "{:4.4f} ".format(recalls[i])
        pre_line += "{:4.4f} ".format(precisions[i])
        f1s_line += "{:4.4f} ".format(f1s[i])
        crs_line += "{:4.4f} ".format(crs[i])

    f.write("\nThe recalls:\n")
    f.write(rec_line+"\n")
    f.write("\nThe precisions:\n")
    f.write(pre_line+"\n")
    f.write("\nThe f1s:\n")
    f.write(f1s_line+"\n")
    f.write("\nThe crs:\n")
    f.write(crs_line+"\n")
