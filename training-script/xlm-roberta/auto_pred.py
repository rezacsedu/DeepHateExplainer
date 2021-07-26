import torch
import torch.nn as nn
import numpy as np
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
def loss_fn(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1))

def get_prediction_value(data_loader, model, device):
        model.eval()
        real_values = []
        fin_outputs = []
        with torch.no_grad():
            for bi, d in enumerate(data_loader):
                ids = d["ids"]
                mask = d["mask"]
                targets = d["targets"]
                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)

                outputs = model(
                    ids=ids,
                    attention_mask=mask
                )
                outputs = torch.softmax(outputs,dim=1)
                outputs = torch.argmax(outputs,dim=1)
                real_values.extend(targets)
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
        real_values = torch.stack(real_values).cpu()
        return fin_outputs, real_values