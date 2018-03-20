import os
from time import ctime, sleep
import threading
from sklearn.externals import joblib
""" --------------------------User Configure----------------------------------------"""
import get_predict_data
""" Example

def get_predict_data(int index):
    #params: predict data index 
    return data
"""

class DataConfig(object):
    # Take a look: three variables are defined by yourself
    # params line_range : the range of the line for predict
    # params predict_nb : the number of sample for predicting every thread
    predict_nb = 1000000
    feature_nb = 76
    line_range = 1641   
""" --------------------------------------------------------------------------------"""

mainpath = os.getcwd()  

def check_folder(filepath):
    """
    check whether filepath exists.
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath

""" 模型路径 """
model_path = os.path.join(mainpath,"Shallowmodel/SVC.model")
""" 预测结果保存路径 """
predict_result_filepath = check_folder(os.path.join(mainpath, "Results/point_to_label/SVM/predict_result"))

""" 用于多线程预测的全局变量 """


thread_nb = 30
lines_label = np.ones((thread_nb, DataConfig.predict_nb), dtype='float32')  

def model_multithread_predict(index,model,data):
    """
    svc predict.
    """
    global lines_label
    print('predict line:{}, time: {}'.format(index,ctime()))
    temp = model.predict_proba(data[:,:])
    lines_label[index,:]  = temp[:,1]


def svc_predict_all():
    """
    svc model predict.
    params start : start_line for predict

    """
    start = 0

    """ load mmodel """
    model = joblib.load(model_path)

    """ read corr file """
    corr = read_corr()
    
    """ beign predict """
    start_time = ctime()
    print('Begin_{}'.format(start))
    
    
    label = np.empty((DataConfig.line_range, DataConfig.predict_nb), dtype='float32')
    lines_data = np.zeros((thread_nb,DataConfig.predict_nb,DataConfig.feature_nb), dtype='float32')

    # 读取地震属性数据，trace
    thread_range = DataConfig.line_range // thread_nb
    rest_line = DataConfig.line_range % thread_nb
    
    for line in range(start,start+thread_range,thread_nb):#DataConfig.line_nb
        print("reading seismic data : {}".format(ctime()))
        for thread in range(thread_nb):
            lines_data[thread,:,:] = get_predict_data(line+thread)

        """ multi threads predicting """
        print("multi threads predicting: {}".format(ctime()))
        threads = []
        for i in range(thread_nb):
            t = threading.Thread(target=model_multithread_predict, args=(i,model,lines_data[i,:,:]))
            threads.append(t)
        for t in threads:
            t.setDaemon(True)
            t.start()
            
        for t in threads:
            t.join()#会阻塞主线程,等待子线程全部跑完再继续下一步，不然值就不会写入，因为主线程不等待。

        print("Round time:{}".format(ctime()))
        for i in range(thread_nb):
            label[line+i,:,:] = lines_label[i,:]
    
    end_time = ctime()
    print('start_time:{} and end_time:{}'.format(start_time,end_time))







