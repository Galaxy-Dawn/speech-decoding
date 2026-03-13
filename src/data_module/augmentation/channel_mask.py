import torch


class ChannelMasking:
    def __init__(self, mask_prob=0.2):
        self.mask_prob = mask_prob

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Apply channel masking to the input batch of images.
        Args:
            imgs (torch.Tensor): Input tensor with shape [batch, channel, timestep]
        Returns:
            torch.Tensor: Masked tensor
        """
        batch_size, num_channels, _ = imgs.size()
        mask = torch.rand(batch_size, num_channels, 1, device=imgs.device) < self.mask_prob
        imgs = imgs.clone()
        imgs[mask.expand_as(imgs)] = 0
        return imgs