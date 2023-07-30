import torchvision
import torch



def plot_to_tensorboard(writer, loss_critic, loss_gen,real,fake, tb_step,images=False):
    writer.add_scalar('loss_critic', loss_critic, tb_step)
    writer.add_scalar('loss_gen', loss_gen, tb_step)


    if images:

        with torch.no_grad():
            img_grid_fake = torchvision.utils.make_grid(fake[:4], normalize=True) # added [1] so we plot the reconstructed phase not magnitude
            img_grid_real = torchvision.utils.make_grid(real[:4], normalize=True)

            writer.add_image('img_grid_fake', img_grid_fake, tb_step)
            writer.add_image('img_grid_real', img_grid_real, tb_step)