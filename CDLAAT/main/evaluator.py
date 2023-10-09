from tqdm.notebook import tqdm
from util import *
from collections import OrderedDict
from dataloaders import TextDataLoader, TextDataset
class Evaluator:
    def __init__(self,model: torch.nn.Module,vocab,criterions,n_training_labels,device):
        self.model = model
        self.vocab = vocab
        self.index_to_label = vocab.index2label
        self.multilabel = True
        self.criterions = criterions
        self.n_training_labels = n_training_labels
        self.device = device

    def evaluate(self,dataloader: TextDataLoader) -> dict:
        self.model.eval()
        pred_probs = []
        true_labels = []
        ids = []
        losses = []
        all_loss_list = []

        for text_batch, label_batch, length_batch, id_batch, desc_batch in tqdm(dataloader, unit="batches", desc="Evaluating"):
            text_batch = text_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            length_batch = length_batch.to(self.device)
            desc_batch = desc_batch.to(self.device)
            true_label_batch = label_batch.cpu().numpy()
            true_labels.extend(true_label_batch)
            ids.extend(id_batch)
            with torch.no_grad():
                output = self.model(text_batch, length_batch,desc_batch)
            loss_list = self.criterions(output, label_batch)
            all_loss_list.append([loss_list.item()])
            probs = [None] * len(output)
            output = torch.sigmoid(output)
            output = output.detach().cpu().numpy()
            probs = output.tolist()
            pred_probs.extend(output)
            loss = get_loss(loss_list, self.n_training_labels)
            losses.append(loss.item())

        scores = OrderedDict()
        scores = calculate_eval_metrics(ids, true_labels,pred_probs, self.multilabel)
        scores["loss"] = np.mean(all_loss_list).item()
        scores["average"] = np.mean(losses).item()
        scores["pred_probs"] = pred_probs
        scores["true_labels"] = true_labels
        scores["hadm_ids"] = ids

        return scores