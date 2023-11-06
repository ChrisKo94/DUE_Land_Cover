import numpy as np

import torch
import torch.nn.functional as F

import gpytorch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from .datasets import get_dataset


def prepare_ood_datasets(true_dataset, ood_dataset):
    #ood_dataset.transform = true_dataset.transform

    datasets = [true_dataset, ood_dataset]

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    return dataloader, anomaly_targets


def loop_over_dataloader(model, likelihood, dataloader):
    model.eval()
    if likelihood is not None:
        likelihood.eval()

    with torch.no_grad():
        scores = []
        scores_dsm = []
        scores_msp = []
        accuracies = []
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()

            if likelihood is None:
                logits = model(data)
                output = F.softmax(logits, dim=1)

                # Dempster-Shafer uncertainty for SNGP
                # From: https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/ood_utils.py#L22
                num_classes = logits.shape[1]
                belief_mass = logits.exp().sum(1)
                uncertainty = num_classes / (belief_mass + num_classes)
            else:
                with gpytorch.settings.num_likelihood_samples(32):
                    y_pred = model(data).to_data_independent_dist()
                    output = likelihood(y_pred).probs.mean(0)

                uncertainty = -(output * output.log()).sum(1)
                uncertainty_dsm = output.shape[1] / (likelihood(y_pred).logits.mean(0).exp().sum(1) + output.shape[1])
                msp, _ = torch.max(output, 1)
                uncertainty_msp = 1 - msp

            pred = torch.argmax(output, dim=1)
            accuracy = pred.eq(target)

            accuracies.append(accuracy.cpu().numpy())
            scores.append(uncertainty.cpu().numpy())
            scores_dsm.append(uncertainty_dsm.cpu().numpy())
            scores_msp.append(uncertainty_msp.cpu().numpy())

    scores = np.concatenate(scores)
    scores_dsm = np.concatenate(scores_dsm)
    scores_msp = np.concatenate(scores_msp)
    accuracies = np.concatenate(accuracies)

    return scores, scores_dsm, scores_msp, accuracies


def get_ood_metrics(in_dataset, out_dataset, model, likelihood=None, root="./"):
    _, _, _, in_dataset = get_dataset(in_dataset, root=root)
    _, _, _, out_dataset = get_dataset(out_dataset, root=root)

    dataloader, anomaly_targets = prepare_ood_datasets(in_dataset, out_dataset)

    scores, scores_dsm, scores_msp, accuracies = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies[: len(in_dataset)])
    auroc = roc_auc_score(anomaly_targets, scores)

    precision, recall, _ = precision_recall_curve(anomaly_targets, scores)
    aupr = auc(recall, precision)

    auroc_dsm = roc_auc_score(anomaly_targets, scores_dsm)

    precision_dsm, recall_dsm, _ = precision_recall_curve(anomaly_targets, scores_dsm)
    aupr_dsm = auc(recall_dsm, precision_dsm)

    auroc_msp = roc_auc_score(anomaly_targets, scores_msp)

    precision_msp, recall_msp, _ = precision_recall_curve(anomaly_targets, scores_msp)
    aupr_msp = auc(recall_msp, precision_msp)

    return accuracy, auroc, aupr, auroc_dsm, aupr_dsm, auroc_msp, aupr_msp


def get_auroc_classification(data, model, likelihood=None):
    if isinstance(data, torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        dataloader = data

    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)

    return accuracy, roc_auc
