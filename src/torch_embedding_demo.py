import torch


def main():
    embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=3)
    words = [[2, 3], [1, 2]]
    embed = embedding(torch.LongTensor(words))
    print(embed)
    print(embed.size())


if __name__ == "__main__":
    main()


