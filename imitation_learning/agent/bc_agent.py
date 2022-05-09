import torch
from agent.networks import CNN

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class BCAgent:
    
    def __init__(self,lr=.01):
        # TODO: Define network, loss function, optimizer
        self.net = CNN()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        pass

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize

        net = self.net.to(device)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # print(type(X_batch))
        # print(X_batch.shape)
        net.train()
        # TODO: forward + backward + optimize
        running_loss = 0.0
        # for i in range(len(X_batch)):
        #     # print(inputs.size())
        #     inputs = X_batch[i].to(device)
        #     labels = y_batch[i].to(device)
        # print(type(inputs))
        # print(inputs.shape)
        self.optimizer.zero_grad()
        outputs = net(X_batch)
        # print(outputs.dtype)
        # print(y_batch.dtype)
        # print(outputs.size())
        # print(y_batch.size())
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()
        running_loss = loss.item()
        # net.eval()
        # with torch.no_grad():
        #     output = net(X_batch)
        #     loss = self.criterion(output, y_batch)
        # loss = loss.numpy()
        return running_loss

    def predict(self, X):
        # TODO: forward pass
        net = self.net.to(device)
        net.eval()
        with torch.no_grad():
            # for inputs in X:
            X = X.to(device)
            outputs = net(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

