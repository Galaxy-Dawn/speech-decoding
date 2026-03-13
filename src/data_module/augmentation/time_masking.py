import torch


class TimeMasking:
    def __init__(self, mask_ratio=0.05, num_masks=4):
        """
        Initialize time masking.
        Args:
            mask_ratio (float): Ratio of timesteps to mask.
            num_masks (int): Number of masks to apply per sample.
        """
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to the input batch of images.
        Args:
            imgs (torch.Tensor): Input tensor with shape [batch, channel, timestep]
        Returns:
            torch.Tensor: Masked tensor
        """
        batch_size, _, timestep = imgs.size()
        mask_param = int(timestep * self.mask_ratio)
        imgs = imgs.clone()
        for i in range(batch_size):
            for _ in range(self.num_masks):
                mask_length = torch.randint(0, mask_param, (1,)).item()
                if mask_length == 0:
                    continue
                start = torch.randint(0, max(timestep - mask_length, 1), (1,)).item()
                imgs[i, :, start:start + mask_length] = 0
        return imgs