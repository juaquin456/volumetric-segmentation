from mdls import Unet3D
from losses import generalized_dice
model = Unet3D.UNet3D(4, 4)
criterion = generalized_dice.GeneralizedDiceLoss(4, False) 