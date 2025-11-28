import torch
import torchaudio.transforms as T
import torch.nn as nn
augmentation= nn.Sequential(
    T.FrequencyMasking(freq_mask_param=15),
    T.TimeMasking(time_mask_param=30)       
)

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    total_loss=0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device, dtype=torch.long)
        
        ##calculate loss
        inputs = augmentation(inputs) ##
        predictions= model(inputs)
        loss= loss_fn(predictions,targets)

        ##backprop
        optimiser.zero_grad() ##to reset the gradients to zero
        loss.backward()      ## claculate weights
        optimiser.step()     ##updates weights
        total_loss+= loss.item()
    print(f"loss: {total_loss/len(data_loader)}")
    return total_loss/len(data_loader)

def train(model, train_data_loader, loss_fn, optimiser, device, epochs, early_stopper, val_data_loader):
    train_loss_history= []
    val_loss_history= []
    best_val_loss= float('inf')
    model= model.to(device)
    for i in range(epochs):
        print(f"==========================\nepoch {i+1}")
        train_loss= train_one_epoch(model, train_data_loader, loss_fn, optimiser, device)
        val_loss= validate(model, val_data_loader, loss_fn, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss= val_loss
            torch.save(model.state_dict(), "Audio_CNN.pth")
        
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early Stopping Triggered")
            break
    print("train complete :)")
    return train_loss_history, val_loss_history

def validate(model, loader, loss, device):
    model.eval()
    model.to(device)
    total_loss= 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels= inputs.to(device), labels.to(device)
            preds= model(inputs)
            loss_value= loss(preds, labels)
            total_loss+= loss_value.item()
    print(f"Validation Loss: {total_loss / len(loader)}")
    return total_loss / len(loader)

class EarlyStopping:
        def __init__(self, patience=10, min_delta=0):
            self.patience= patience
            self.min_delta= min_delta
            self.best_loss= None
            self.counter= 0
            self.early_stop= False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss= val_loss
                return
                
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss= val_loss
                self.counter= 0
            else:
                self.counter+= 1
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
    
                if self.counter >= self.patience:
                    self.early_stop= True