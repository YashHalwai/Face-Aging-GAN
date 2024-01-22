# Import necessary libraries
import itertools
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

# Import custom modules
from dataset import ImagetoImageDataset
from models import Generator, Discriminator

# Define a LightningModule class named AgingGAN
class AgingGAN(pl.LightningModule):

    # Initialize the class with hyperparameters
    def __init__(self, hparams):
        super(AgingGAN, self).__init__()

        # Save hyperparameters for easy access
        self.save_hyperparameters(hparams)

        # Initialize generator and discriminator models
        self.genA2B = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.genB2A = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.disGA = Discriminator(hparams['ndf'])
        self.disGB = Discriminator(hparams['ndf'])

        # Cache for generated images
        self.generated_A = None
        self.generated_B = None
        self.real_A = None
        self.real_B = None

    # Forward pass of the generator
    def forward(self, x):
        return self.genA2B(x)

    # Training step for the LightningModule
    def training_step(self, batch, batch_idx, optimizer_idx):
        # Extract real images from the batch
        real_A, real_B = batch

        if optimizer_idx == 0:  # Generator training
            # Identity loss
            same_B = self.genA2B(real_B)
            loss_identity_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']

            same_A = self.genB2A(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

            # GAN loss
            fake_B = self.genA2B(real_A)
            pred_fake = self.disGB(fake_B)
            loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * self.hparams['adv_weight']

            fake_A = self.genB2A(real_B)
            pred_fake = self.disGA(fake_A)
            loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * self.hparams['adv_weight']

            # Cycle loss
            recovered_A = self.genB2A(fake_B)
            loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.hparams['cycle_weight']

            recovered_B = self.genA2B(fake_A)
            loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.hparams['cycle_weight']

            # Total loss
            g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            # Output dictionary for logging
            output = {
                'loss': g_loss,
                'log': {'Loss/Generator': g_loss}
            }
            self.log('Loss/Generator', g_loss)

            # Cache generated and real images for logging
            self.generated_B = fake_B
            self.generated_A = fake_A
            self.real_B = real_B
            self.real_A = real_A

            # Log images to tensorboard every 500 batches
            if batch_idx % 500 == 0:
                self.genA2B.eval()
                self.genB2A.eval()
                fake_A = self.genB2A(real_B)
                fake_B = self.genA2B(real_A)
                self.logger.experiment.add_image('Real/A', make_grid(self.real_A, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.logger.experiment.add_image('Real/B', make_grid(self.real_B, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.logger.experiment.add_image('Generated/A',
                                                 make_grid(self.generated_A, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.logger.experiment.add_image('Generated/B',
                                                 make_grid(self.generated_B, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.genA2B.train()
                self.genB2A.train()

            return output

        if optimizer_idx == 1:  # Discriminator training
            # Real loss for discriminator GA
            pred_real = self.disGA(real_A)
            loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

            # Fake loss for discriminator GA
            fake_A = self.generated_A
            pred_fake = self.disGA(fake_A.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

            # Total loss for discriminator GA
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            # Real loss for discriminator GB
            pred_real = self.disGB(real_B)
            loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

            # Fake loss for discriminator GB
            fake_B = self.generated_B
            pred_fake = self.disGB(fake_B.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

            # Total loss for discriminator GB
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            # Total discriminator loss
            d_loss = loss_D_A + loss_D_B
            output = {
                'loss': d_loss,
                'log': {'Loss/Discriminator': d_loss}
            }
            self.log('Loss/Discriminator', d_loss)

            return output

    # Configure optimizers for the LightningModule
    def configure_optimizers(self):
        # Define optimizers for generator and discriminator
        g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(),
                                                   self.disGB.parameters()),
                                   lr=self.hparams['lr'],
                                   betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        return [g_optim, d_optim], []

    # DataLoader for training
    def train_dataloader(self):
        # Define image transformations for training data
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
            transforms.RandomCrop(self.hparams['img_size']),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            #transforms.RandomPerspective(p=0.5),
            transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        # Create dataset using the defined transformations
        dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], train_transform)
        
        # Create DataLoader with specified batch size and number of workers
        return DataLoader(dataset,
                          batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'],
                          shuffle=True)


# Reason 1: Simplicity and Generality
# CycleGAN is known for its simplicity and general applicability. It doesn't require paired data during training, making it versatile for various image translation tasks.
# CAAE, on the other hand, may need paired data during training, limiting its applicability in scenarios where obtaining paired data is challenging or expensive.

# Reason 2: Unpaired Image Translation
# CycleGAN excels in unpaired image translation, where there is no one-to-one correspondence between samples in the source and target domains.
# CAAE, being a conditional model, may struggle in scenarios where obtaining perfectly paired samples is not feasible, hindering its performance.

# Reason 3: Cycle Consistency for Image-to-Image Translation
# CycleGAN enforces cycle consistency, meaning that translating an image from domain A to B and then back from B to A should ideally result in the original image.
# This cycle consistency loss helps to generate realistic translations and maintain visual consistency, a crucial aspect for image-to-image translation tasks.

# Reason 4: Reduced Mode Collapse
# CycleGAN is less prone to mode collapse compared to some other GAN variants. Mode collapse occurs when the generator collapses to a limited set of outputs, resulting in a lack of diversity in generated samples.
# CAAE might face challenges in mitigating mode collapse, affecting the variety and quality of generated samples.

# Reason 5: Training Stability
# CycleGAN training is often more stable due to the use of cycle consistency, adversarial loss, and identity loss. These components contribute to a more balanced training process.
# CAAE, depending on the specifics of its architecture, may face challenges related to training stability, potentially leading to convergence issues.

# Reason 6: No Need for Conditional Information
# In many image translation tasks, the relationship between input and output can be learned without explicit conditional information.
# CycleGAN does not require additional conditional input, simplifying the architecture and training process.
# CAAE, being a conditional model, may introduce complexity by requiring additional conditional information for effective training.

# Reason 7: Better Handling of Complex Transformations
# CycleGAN is often preferred when dealing with complex transformations where the mapping between source and target domains is not straightforward.
# CAAE might struggle in scenarios where the conditional relationship is intricate, making it challenging to model complex mappings effectively.

# Conclusion: 
# While CAAE can be effective in certain scenarios, CycleGAN's simplicity, ability to handle unpaired data, enforce cycle consistency, and exhibit training stability make it a preferred choice for many image-to-image translation tasks.
# The choice between these models should be based on the specific requirements and characteristics of the dataset and task at hand.
