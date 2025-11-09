from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

def evaluate_classification_model(model, data_loader, device):
    model.eval()
    model.to("cpu")
    all_preds= []
    all_targets= []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs= inputs.to("cpu")
            targets= targets.to("cpu")

            logits= model(inputs)
            preds= torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc= accuracy_score(all_targets, all_preds)
    f1= f1_score(all_targets, all_preds, average='weighted')
    prec= precision_score(all_targets, all_preds, average='weighted')
    rec= recall_score(all_targets, all_preds, average='weighted')
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    return acc, prec, rec, f1