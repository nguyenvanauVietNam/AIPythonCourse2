# model_utils.py

import torch
from torch import nn, optim
from torchvision import models

def load_model(arch='vgg13', hidden_units=512):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        return model
    else:
        raise ValueError(f"Unsupported architecture {arch}")

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    
    return model


def train_model(model, trainloader, validloader, epochs, learning_rate, gpu):
    # Define the loss criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validate model after each epoch
        valid_loss, accuracy = validate_model(model, validloader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")
        
def validate_model(model, validloader, criterion, device):
    # Function to validate model performance on validation dataset
    model.eval()
    valid_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return valid_loss, accuracy

def save_checkpoint(model, save_dir, arch, hidden_units, learning_rate, epochs):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier
    }
    
    torch.save(checkpoint, f"{save_dir}/checkpoint_resnet50.pth")


def load_checkpoint(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    model = load_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def predict(image, model, topk=5, device='cpu'):
    model.eval()
    model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        logps = model(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    
    top_p = top_p.cpu().numpy()[0]
    top_class = top_class.cpu().numpy()[0]
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[c] for c in top_class]
    
    return top_p, top_class
