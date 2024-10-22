import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from model import Model
from dataset import LungCancerDataset


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((1,), (1,))
])
dataset = LungCancerDataset(transform)

epochs = 5
batch_size = 32
learning_rate = 0.001

training_split = int(0.75 * len(dataset))
testing_split = len(dataset) - training_split
training_data, testing_data = random_split(dataset, [training_split, testing_split])
training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

losses = []
for i in range(epochs):
    for images, labels in training_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    print(loss.item())

correct = 0
model.eval()
with torch.no_grad():
    for images, labels in testing_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        _, predicted = torch.max(predictions.data, 1)
        correct += (predicted == labels).sum().item()

print((correct / testing_split) * 100)
