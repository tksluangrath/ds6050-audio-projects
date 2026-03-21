import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AST(nn.Module):
    def __init__(
        self,
        num_classes=12,
        n_mels=128,
        time_frames=101,
        patch_size=16,
        fstride=10,
        tstride=10,
    ):
        super().__init__()

        deit = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)

        # Patch embedding: adapt 3-channel DeiT projection to 1-channel
        old_proj = deit.patch_embed.proj  # Conv2d(3, 768, kernel_size=16, stride=16)
        self.patch_embed = nn.Conv2d(
            1, 768, kernel_size=patch_size, stride=(fstride, tstride)
        )
        with torch.no_grad():
            self.patch_embed.weight.copy_(
                old_proj.weight.mean(dim=1, keepdim=True)
            )
            self.patch_embed.bias.copy_(old_proj.bias)

        # Compute output grid size
        n_freq = (n_mels - patch_size) // fstride + 1      # (128-16)//10 + 1 = 12
        n_time = (time_frames - patch_size) // tstride + 1  # (101-16)//10 + 1 = 9

        # Interpolate positional embeddings from DeiT's 14x14 grid to n_freq x n_time
        # DeiT-distilled pos_embed shape: (1, 198, 768) = 1 CLS + 1 distill + 196 patches
        old_pos = deit.pos_embed
        cls_pos = old_pos[:, :1, :]      # (1, 1, 768)
        patch_pos = old_pos[:, 2:, :]    # (1, 196, 768) — skip distillation token at index 1
        patch_pos = patch_pos.reshape(1, 768, 14, 14).float()
        patch_pos = F.interpolate(
            patch_pos, size=(n_freq, n_time), mode="bilinear", align_corners=False
        )
        patch_pos = patch_pos.flatten(2).transpose(1, 2)    # (1, n_patches, 768)
        self.pos_embed = nn.Parameter(
            torch.cat([cls_pos, patch_pos], dim=1)           # (1, 1+n_patches, 768)
        )

        # CLS token
        self.cls_token = nn.Parameter(deit.cls_token.data.clone())

        # Transformer encoder blocks and layer norm
        self.blocks = deit.blocks
        self.norm = deit.norm

        # Classification head
        self.head = nn.Linear(768, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        del deit

    def forward(self, x):
        # x: (B, 1, n_mels, time_frames)
        x = self.patch_embed(x)              # (B, 768, n_freq, n_time)
        x = x.flatten(2).transpose(1, 2)    # (B, n_patches, 768)

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      # (B, 1+n_patches, 768)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])
