

""" --------------------------User Configure----------------------------------------"""
import get_dataset
""" Example
def get_dataset():
    return train_x, train_y, test_x, test_y
"""
""" --------------------------------------------------------------------------------"""


import numpy as np
import csv
from sklearn.svm import SVC
#from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os

mainpath = os.getcwd()

""" 测试结果保存路径 """
test_result_filepath = check_folder(os.path.join(mainpath,"test_result"))
""" 测试结果图保存路径 """
test_presentation_filepath = check_folder(os.path.join(mainpath,"test_presentation"))
""" 模型保存路径 """
model_path = os.path.join(mainpath,"Shallowmodel/SVC.model")

def check_folder(filepath):
    """
    check whether filepath exists.
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath

def svc():
    """ 
    Train and choose best params. 
    """

    #Take a look: get_dataset() is defined by yourself
    #train_x : [None, feature_nb]
    #train_y : [None, 1]
    train_x, train_y, test_x, test_y = get_dataset()


    params = {'C':[0.1,1,10,100,1000], 'gamma':[0.0001, 0.001],'kernel':['rbf']}
    gs_all = GridSearchCV(estimator = SVC(probability=True), param_grid=params, scoring='neg_log_loss', cv=5, n_jobs=20)
    gs_all.fit(train_x, np.ravel(train_y))


    """ train best_params' model """
    print(gs_all.grid_scores_)
    print(gs_all.best_params_)
    print(gs_all.best_score_)

    c = gs_all.best_params_["C"]
    g = gs_all.best_params_["gamma"]
    k = gs_all.best_params_["kernel"]
    best_svc = SVC(C=c, gamma=g, kernel=k, probability=True)
    best_svc.fit(train_x, np.ravel(train_y))

    """ save best model """
    joblib.dump(best_svc, model_path)

    """ predict test_data """
    pred = best_svc.predict_proba(test_x)

    """ save predict result to csv """
    os.chdir(test_result_filepath)
    with open('pre_label.csv','w') as file:
        write = csv.writer(file)
        for i in range(len(test_x)):
            row = []
            row.append(pred[i][1])
            row.append(test_y[i])
            write.writerow(row)

    """ metric evaluate """

    os.chdir(test_result_filepath)
    with open('evaluate_metrics.csv','w') as file:
        writer = csv.writer(file,lineterminator='\n')
        writer.writerow(['Threshold','TP','TN','FP','FN','precision','recall','FDR','TDR'])
        for i in range(200):
            threshold = i/199
            evaulate(threshold)
            (TP,TN,FP,FN),(precision,recall),(FPR,TPR) = calc_metrics()
            writer.writerow([threshold,TP,TN,FP,FN,precision,recall,FPR,TPR])

    """ plot PR and ROC Curve """
    evaluate_plot()
    

def convert(data,threshold):
    conv_data = []
    if float(data[0]) >= threshold:
        conv_data.append(1)
        conv_data.append(int(float(data[1])))
    else:
        conv_data.append(0)
        conv_data.append(int(float(data[1])))
    return conv_data

def evaulate(threshold, filename='pre_label.csv'):
    with open(filename) as readfile,open('process_'+filename,'w') as writefile:
        writer = csv.writer(writefile,lineterminator='\n')
        content = csv.reader(readfile)
        for i,line in enumerate(content):
            data = convert(line,threshold)
            writer.writerow(data)

def calc_metrics(filename='process_pre_label.csv'):
    epsilon = 0.1
    TP,FP,TN,FN = (0,0,0,0)
    Acc = 0
    nb = 0
    with open(filename) as file:
        content = csv.reader(file)
        for data in content:
            nb += 1
            if int(data[0]) == 1:
                if int(data[1]) == 1:
                    Acc += 1
                    TP += 1
                else:
                    FP += 1
            else:
                if int(data[1]) == 0:
                    Acc += 1
                    TN += 1
                else:
                     FN += 1
    precision = (epsilon+TP)/(epsilon+FP+TP)
    recall = (epsilon+TP)/(epsilon+TP+FN)
    FPR = (epsilon+FP)/(epsilon+FP+TN)
    TPR = (epsilon+TP)/(epsilon+TP+FN)
    return (TP,TN,FP,Acc/nb),(precision,recall),(FPR,TPR)



def evaluate_plot():
    num = 1
    Precision = np.empty((num,200),dtype='float32')
    Recall = np.empty((num,200),dtype='float32')
    FPR = np.empty((num,200),dtype='float32')
    TPR = np.empty((num,200),dtype='float32')
    plt.figure(1,figsize=(15,10))
    
    os.chdir(test_presentation_filepath)
    j = 0
    with open(filename,'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if line[0] != 'Threshold':
                Precision[i][j] = float(line[5])
                Recall[i][j] = float(line[6])
                FPR[i][j] = float(line[7])
                TPR[i][j] = float(line[8])
                j+=1
    
    os.chdir(test_presentation_filepath)

    """ plot ROC Curve """
    plt.subplot(1,1,1)
    plt.plot(FPR[i,:],TPR[i,:],label=label)
    plt.title('SVC ROC Curve',fontsize=15)
    plt.xlabel('FPR',fontsize=15)
    plt.ylabel('TPR',fontsize=15)
    legend = plt.legend(loc='center left',bbox_to_anchor=(0.85,0.3),borderpad=0.1,labelspacing=0.1)
    plt.savefig('SVC_ROC.png',dpi=500)
    plt.close('all')

    """ plot PR Curve """
    plt.subplot(1,1,1)
    plt.plot(Recall[i,:],Precision[i,:],label=label)
    plt.title('SVC PR Curve',fontsize=15)
    plt.xlabel('Recall',fontsize=15)
    plt.ylabel('Precision',fontsize=15)
    legend = plt.legend(loc='center left',bbox_to_anchor=(0.85,0.7),borderpad=0.1,labelspacing=0.1)
    plt.savefig('SVC_PR.png',dpi=500) 

if __name__ == '__main__':
    svc()