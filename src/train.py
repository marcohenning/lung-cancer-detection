import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from model import Model
from dataset import LungCancerDataset
from prettytable import PrettyTable


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((1,), (1,))
])
dataset = LungCancerDataset(transform)

epochs = 40
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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
losses_table = PrettyTable()
losses_table.field_names = ['Epoch', 'Loss']

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
    if i == 0 or (i + 1) % 2 == 0:
        losses_table.add_row([i + 1, f'{loss.item():.10f}'])

print(losses_table)

correct = 0
model.eval()
with torch.no_grad():
    for images, labels in testing_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        _, predicted = torch.max(predictions.data, 1)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {(correct / testing_split * 100):.2f}% ({correct}/{testing_split})')

plt.get_current_fig_manager().set_window_title('Training')
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
