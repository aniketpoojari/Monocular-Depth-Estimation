import torch
import torch.nn as nn
import timm


class ReassembleLayer(nn.Module):
    def __init__(self, embed_dim, scale, decoder_channels):
        super(ReassembleLayer, self).__init__()
        self.decoder_channels = decoder_channels
        self.embed_dim = embed_dim
        self.scale = scale
        self.project = nn.Conv2d(
            self.embed_dim, self.decoder_channels, kernel_size=1, stride=1, padding=0
        )
        if scale == 4:
            # Upsample 4x: bilinear + conv (avoids checkerboard from ConvTranspose2d)
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
                nn.Conv2d(self.decoder_channels, self.decoder_channels, kernel_size=3, padding=1),
            )
        elif scale == 8:
            # Upsample 2x
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(self.decoder_channels, self.decoder_channels, kernel_size=3, padding=1),
            )
        elif scale == 16:
            # No spatial change
            self.resample = nn.Conv2d(
                self.decoder_channels, self.decoder_channels,
                kernel_size=3, stride=1, padding=1,
            )
        else:
            # Downsample 2x
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
                nn.Conv2d(self.decoder_channels, self.decoder_channels, kernel_size=3, padding=1),
            )

    def forward(self, tokens):
        cls = tokens[:, 0, :]
        tokens = tokens[:, 1:, :]
        b, n, d = tokens.size()
        h = w = int(n**0.5)
        cls = cls.view(-1, 1, cls.shape[1])
        cls = cls.expand(-1, n, -1)
        tokens = tokens + cls

        x = tokens.permute(0, 2, 1).reshape(b, self.embed_dim, h, w)
        x = self.project(x)
        x = self.resample(x)

        return x


class FusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(FusionBlock, self).__init__()
        self.resconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, skip_connection):
        x = x + skip_connection
        x = self.resconv(x)
        x = self.upsample(x)
        return x


class DPT(nn.Module):

    def __init__(self, decoder_channels=256):
        super().__init__()
        self.decoder_channels = decoder_channels

        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=1
        )

        self.reassemble_layers = nn.ModuleList(
            [
                ReassembleLayer(embed_dim=768, scale=4, decoder_channels=decoder_channels),
                ReassembleLayer(embed_dim=768, scale=8, decoder_channels=decoder_channels),
                ReassembleLayer(embed_dim=768, scale=16, decoder_channels=decoder_channels),
                ReassembleLayer(embed_dim=768, scale=32, decoder_channels=decoder_channels),
            ]
        )

        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(decoder_channels) for _ in range(4)]
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_channels // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        tokens = self.vit.patch_embed(x)

        cls_token = self.vit.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens += self.vit.pos_embed

        specific_tokens = []
        for i, layer in enumerate(self.vit.blocks):
            tokens = layer(tokens)
            if (i + 1) % 3 == 0:
                specific_tokens.append(tokens)

        for i in range(len(specific_tokens)):
            specific_tokens[i] = self.vit.norm(specific_tokens[i])

        reassembled_features = []
        for i, reassemble_layer in enumerate(self.reassemble_layers):
            reassembled_features.append(reassemble_layer(specific_tokens[i]))
        reassembled_features.append(
            torch.zeros(b, self.decoder_channels, 7, 7, device=x.device)
        )

        x = reassembled_features[-1]
        for i in range(len(self.fusion_blocks)):
            x = self.fusion_blocks[i](x, reassembled_features[-(i + 2)])

        x = self.output_head(x)

        return x
