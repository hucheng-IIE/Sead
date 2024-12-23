import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, embed_dim, lr):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

        self.optimizer_D = torch.optim.Adam(self.parameters(), lr=lr)
        self.discriminator_loss = nn.BCELoss()

    def forward(self, embedding):
        result = self.classifier(embedding)
        return result
 
    def update(self, real_embeddings, fake_embeddings):
        self.optimizer_D.zero_grad()
        #(batch_size,num_rels,dim)
        real_labels = torch.ones(real_embeddings.size(0), real_embeddings.size(1), 1).cuda()
        fake_labels = torch.zeros(fake_embeddings.size(0), fake_embeddings.size(1), 1).cuda()

        real_validity = self.forward(real_embeddings)
        real_loss = self.discriminator_loss(real_validity, real_labels)

        fake_validity = self.forward(fake_embeddings)
        fake_loss = self.discriminator_loss(fake_validity, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward(retain_graph=True)
        self.optimizer_D.step()
        
        return d_loss.item()