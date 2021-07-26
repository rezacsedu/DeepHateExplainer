import utils
import dataset
import engine
import torch
import transformers
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from settings import get_module_logger
from model import RobertaBengali, RobertaBengaliTwo, RobertaBengaliNext, CustomRobertaBengali
from sklearn import model_selection
from transformers import AdamW
from dataset import BengaliDataset
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from collections import defaultdict
from prediction import get_predictions
from auto_pred import get_prediction_value
import gc
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)

def test_evaluation(model, device):
    dfx = pd.read_csv(args.testing_file).dropna().reset_index(drop=True)
    test_dataset = BengaliDataset(
        text=dfx.text.values,
        targets=dfx.target.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=2
    )
    y_pred, y_test = get_prediction_value(test_data_loader, model, device)
    dfx['y_pred'] = y_pred
    pred_test = dfx[['text','target','y_pred']]
    pred_test.to_csv(f'../DeepHateLingo/training-script/xlm-roberta/output/{args.output}',index = False)
    print('Accuracy::', metrics.accuracy_score(y_test, y_pred))
    print('Mcc Score::', matthews_corrcoef(y_test, y_pred))
    print('Precision::', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall::', metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F_score::', metrics.f1_score(y_test, y_pred, average='weighted'))
    print('classification_report:: ', metrics.classification_report(y_test, y_pred))#target_names=["real", "fake"]))
    logger.info('Mcc Score:: {}'.format(matthews_corrcoef(y_test, y_pred)))
    logger.info('Accuracy:: {}'.format(metrics.accuracy_score(y_test, y_pred)))
    logger.info('Precision:: {}'.format(metrics.precision_score(y_test, y_pred, average='weighted')))
    logger.info('Recall:: {}'.format(metrics.recall_score(y_test, y_pred, average='weighted')))
    logger.info('F_score:: {}'.format(metrics.f1_score(y_test, y_pred, average='weighted')))
    logger.info('classification_report:: {}'.format(classification_report(y_test, y_pred)))



if __name__ == "__main__":
    test_evaluation()