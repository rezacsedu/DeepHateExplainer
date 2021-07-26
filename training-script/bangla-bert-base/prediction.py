import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from flag import get_parser
from settings import get_module_logger
from sklearn.metrics import confusion_matrix, classification_report
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)

def get_predictions(df):
    y_test = df.target.values
    y_pred = df.pred.values
    print('Accuracy::', metrics.accuracy_score(y_test, y_pred))
    print('Precision::', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall::', metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F_score::', metrics.f1_score(y_test, y_pred, average='weighted'))
    print('classification_report:: ', metrics.classification_report(y_test, y_pred))#target_names=["real", "fake"]))
    logger.info('Accuracy:: {}'.format(metrics.accuracy_score(y_test, y_pred)))
    logger.info('Precision:: {}'.format(metrics.precision_score(y_test, y_pred, average='weighted')))
    logger.info('Recall:: {}'.format(metrics.recall_score(y_test, y_pred, average='weighted')))
    logger.info('F_score:: {}'.format(metrics.f1_score(y_test, y_pred, average='weighted')))
    logger.info('classification_report:: {}'.format(classification_report(y_test, y_pred)))

