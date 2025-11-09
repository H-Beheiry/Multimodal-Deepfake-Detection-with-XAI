import torch
def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    total_loss=0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device, dtype=torch.long)
        
        ##calculate loss
        predictions= model(inputs)
        loss= loss_fn(predictions,targets)

        ##backprop
        optimiser.zero_grad() ##to reset the gradients to zero
        loss.backward()      ## claculate weights
        optimiser.step()     ##updates weights
        total_loss+= loss.item()
    print(f"loss: {total_loss/len(data_loader)}\n==========================")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    model= model.to(device)
    for i in range(epochs):
        print(f"epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
    print("train completem :)")