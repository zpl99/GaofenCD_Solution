
from sklearn.metrics import precision_recall_fscore_support

"""
confusionMetric  
P\L     P    N
P      TP    FP
N      FN    TN
"""

def test():
    pass
def getPrecision_Recall_F1(truth,predict,label=[0,1]):
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(truth.flatten(), predict.flatten(),
                                                                               labels=label)
    return p_class,r_class,f_class


if __name__ == '__main__':
    test()
