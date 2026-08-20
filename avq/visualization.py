import torch
import matplotlib.pyplot as plt


def image_for_plot(image):
    """
    Prepares a PyTorch image tensor for plotting with Matplotlib.
    """
    image = image.detach().cpu()
    if image.shape[0] == 1:
        return image.squeeze(0)

    return image.permute(1, 2, 0)


@torch.no_grad()
def visual_inspection(model, loader, device, n=8, temperature=0.1, soft_weight=0.1, ste=False):
    """
    Generates a visual comparison of original images alongside their soft and hard reconstructions.
    """
    model.eval()

    imgs, _ = next(iter(loader))
    imgs = imgs[:n].to(device)

    # Images from the loader are already normalized to [-1, 1].
    outputs = model(
        imgs, temperature=temperature, 
        soft_weight=soft_weight, return_separate_reconstructions=True,
    )

    reconstruction_soft = outputs["reconstruction_soft"]
    reconstruction_hard = outputs["reconstruction_hard"]

    # Convert original and reconstructed images from [-1, 1] to [0, 1].
    imgs_display = ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)
    reconstruction_soft_display = ((reconstruction_soft + 1.0) / 2.0).clamp(0.0, 1.0)
    reconstruction_hard_display = ((reconstruction_hard + 1.0) / 2.0).clamp(0.0, 1.0)

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6), squeeze=False)

    for i in range(n):
        original = imgs_display[i]
        soft = reconstruction_soft_display[i]
        hard = reconstruction_hard_display[i]

        for row, image, title in [(0, original, "Original"), (1, soft, "Soft"), (2, hard, "Hard Argmax")]:
            
            cmap = "gray" if image.shape[0] == 1 else None
            vmin = 0.0 if cmap == "gray" else None
            vmax = 1.0 if cmap == "gray" else None

            axes[row, i].imshow(image_for_plot(image), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, i].set_title(title)
            axes[row, i].axis("off")

    title = "VQ-STE" if ste else "AVQ"
    fig.suptitle(f"{title} Reconstructions at Temperature = {temperature:.3f}", fontsize=14)
    plt.tight_layout()
    plt.show()

