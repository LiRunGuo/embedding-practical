import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def prepare_data():
    sentences = [
        "longge like dog",
        "longge like cat",
        "longge like animal",
        "dog cat animal",
        "banana apple cat dog like",
        "dog fish milk like",
        "dog cat animal like",
        "longge like apple",
        "apple like",
        "longge like banana",
        "apple banana longge movie book music like",
        "cat dog hate",
        "cat dog like",
    ]
    word_sequence = " ".join(sentences).split()
    vocab = list(set(word_sequence))
    word2idx = {w: i for i, w in enumerate(vocab)}
    voc_size = len(vocab)

    C = 2
    skip_grams = []
    for idx in range(C, len(word_sequence) - C):
        center = word2idx[word_sequence[idx]]
        context_idx = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        context = [word2idx[word_sequence[i]] for i in context_idx]
        for w in context:
            skip_grams.append([center, w])

    def make_data(pairs):
        input_data = []
        output_data = []
        for center, w in pairs:
            one_hot = np.eye(voc_size)[center]
            input_data.append(one_hot)
            output_data.append(w)
        return torch.Tensor(input_data), torch.LongTensor(output_data)

    input_data, output_data = make_data(skip_grams)
    dataset = Data.TensorDataset(input_data, output_data)
    loader = Data.DataLoader(dataset, batch_size=8, shuffle=True)
    return vocab, voc_size, loader


class Word2Vec(nn.Module):
    def __init__(self, voc_size: int, embedding_size: int = 2):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(voc_size, embedding_size))
        self.W_out = nn.Parameter(torch.randn(embedding_size, voc_size))

    def forward(self, X):
        hidden_layer = torch.matmul(X, self.W_in)
        output_layer = torch.matmul(hidden_layer, self.W_out)
        return output_layer


def train_model():
    vocab, voc_size, loader = prepare_data()
    model = Word2Vec(voc_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(200):
        for i, (batch_x, batch_y) in enumerate(loader):
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"epoch={epoch+1}, loss={loss.item():.4f}")


def main():
    train_model()


if __name__ == "__main__":
    main()


