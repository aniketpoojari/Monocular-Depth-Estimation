import torch
import torch.nn as nn
import timm


class ReassembleLayer(nn.Module):
    def __init__(self, embed_dim, s, new):
        super(ReassembleLayer, self).__init__()
        self.new = new
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(
            self.embed_dim, self.new, kernel_size=1, stride=1, padding=0
        )
        if s == 4:
            self.s = nn.ConvTranspose2d(
                self.new, self.new, kernel_size=3, stride=4, output_padding=1
            )
        elif s == 8:
            self.s = nn.ConvTranspose2d(
                self.new, self.new, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        elif s == 16:
            self.s = nn.Conv2d(self.new, self.new, kernel_size=3, stride=1, padding=1)
        else:
            self.s = nn.Conv2d(self.new, self.new, kernel_size=3, stride=2, padding=1)

    def forward(self, tokens):
        # read
        cls = tokens[:, 0, :]
        tokens = tokens[:, 1:, :]
        b, n, d = tokens.size()
        h = w = int(n**0.5)
        cls = cls.view(-1, 1, cls.shape[1])
        cls = cls.expand(-1, n, -1)
        tokens = tokens + cls
        cls = None

        # concatenate
        x = tokens.permute(0, 2, 1).reshape(b, self.embed_dim, h, w)

        # resample
        x = self.project(x)
        x = self.s(x)

        return x


class FusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(FusionBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, skip_connection):
        x = x + skip_connection
        x = self.upsample(x)
        return x


class DPT(nn.Module):

    def __init__(self, new=256):
        super().__init__()
        self.new = new

        # Define ViT backbone
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=1
        )

        # Reassemble layers to get the feature maps at different scales
        self.reassemble_layers = nn.ModuleList(
            [
                ReassembleLayer(embed_dim=768, s=4, new=new),  # 56
                ReassembleLayer(embed_dim=768, s=8, new=new),  # 28
                ReassembleLayer(embed_dim=768, s=16, new=new),  # 14
                ReassembleLayer(embed_dim=768, s=32, new=new),  # 7
            ]
        )

        self.fusion_blocks = nn.ModuleList(
            [
                FusionBlock(new),
                FusionBlock(new),
                FusionBlock(new),
                FusionBlock(new),
            ]
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(new, new // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(new // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        tokens = self.vit.patch_embed(x)  # Initial embedding

        # Add the [CLS] token to the beginning of the token sequence
        cls_token = self.vit.cls_token.expand(
            b, -1, -1
        )  # Shape: [batch_size, 1, embedding_dim]
        tokens = torch.cat(
            (cls_token, tokens), dim=1
        )  # Shape: [batch_size, num_patches + 1, embedding_dim]

        # Positional embedding
        tokens += self.vit.pos_embed

        specific_tokens = []
        for i, layer in enumerate(self.vit.blocks):
            tokens = layer(tokens)
            if (i + 1) % 3 == 0:
                specific_tokens.append(tokens)

        for i in range(0, len(specific_tokens)):
            specific_tokens[i] = self.vit.norm(specific_tokens[i])

        reassembled_features = []
        for i, reassemble_layer in enumerate(self.reassemble_layers):
            reassembled_features.append(reassemble_layer(specific_tokens[i]))
        reassembled_features.append(torch.zeros([b, self.new, 7, 7]).cuda())

        x = reassembled_features[-1]
        for i in range(len(self.fusion_blocks)):
            x = self.fusion_blocks[i](x, reassembled_features[-(i + 2)])

        x = self.output_head(x)

        return x
