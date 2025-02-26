import random
import pandas as pd
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from thop import profile
from Model.TFSNet import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_data(folder_path):
    labels = []
    matrices = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if 'relaxing' in file_name:
            true_label = 0
        elif 'lifting' in file_name:
            true_label = 1
        elif 'carrying' in file_name:
            true_label = 2
        elif 'playing basketball' in file_name:
            true_label = 3
        elif 'using the mouse' in file_name:
            true_label = 4
        elif 'using the phone' in file_name:
            true_label = 5
        elif 'typing' in file_name:
            true_label = 6
        elif 'writing' in file_name:
            true_label = 7
        elif 'reading' in file_name:
            true_label = 8
        elif 'using chopsticks' in file_name:
            true_label = 9
        elif 'using a spoon' in file_name:
            true_label = 10

        long = 600
        df = pd.read_csv(file_path, header=None)

        if df.isnull().values.any():
            nan_rows = df[df.isnull().any(axis=1)].index.tolist()
            raise ValueError(f"The file {file_name} contains NaN values at rows: {nan_rows}")

        num_samples = int(len(df) // long)

        for i in range(num_samples):
            start_index = i * long
            end_index = (i + 1) * long
            sample_data = df.iloc[start_index:end_index, :].values
            scaler = preprocessing.StandardScaler()
            sample_data = scaler.fit_transform(sample_data)
            matrices.append(sample_data)
            labels.append(true_label)

    X = np.stack(matrices, axis=0)
    Y = pd.get_dummies(labels).values
    return X, Y

def validate(model, criterion, X_valid, Y_valid):
    model.eval()
    with torch.no_grad():
        X_valid_tensor = torch.tensor(X_valid, device=device).float()
        Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long, device=device)
        Y_valid_tensor = torch.argmax(Y_valid_tensor, dim=1)

        output_logits, output_softmax = model(X_valid_tensor)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y_valid_tensor == predictions).item() / len(Y_valid_tensor)
        loss = criterion(output_logits, Y_valid_tensor)

        precision = precision_score(Y_valid_tensor.cpu(), predictions.cpu(), average='macro')
        recall = recall_score(Y_valid_tensor.cpu(), predictions.cpu(), average='macro')
        f1 = f1_score(Y_valid_tensor.cpu(), predictions.cpu(), average='macro')

        return accuracy, precision, recall, f1, loss.item()


if __name__ == '__main__':
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X, Y = load_data(r'Datasets/Fine-Grained Activities Dataset')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(3, 16, 11).to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = r'Experiment Data/Comparison Experiment/Fine-Grained Activities/TFSNet/best.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    accuracy, precision, recall, f1, loss = validate(model, criterion, X_test, Y_test)

    dummy_input = torch.randn(1, *X_train.shape[1:], device=device)
    flops, params = profile(model, inputs=(dummy_input,))

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print(f"Params: {int(params)}")
    print(f"FLOPs: {int(flops)}")
    print(f"Loss: {loss:.3f}")
