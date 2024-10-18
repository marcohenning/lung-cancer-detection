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

training_split = int(0.75 * len(dataset))
testing_split = len(dataset) - training_split
training_data, testing_data = random_split(dataset, [training_split, testing_split])
training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

model = Model()
