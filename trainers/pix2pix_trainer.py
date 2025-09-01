from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
import torch


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 1:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model, device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model.to(f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, _, self.optimizer_D = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data, iter_count):
        self.optimizer_G.zero_grad()
        g_loss_for_backward, g_losses_for_display, generated = self.pix2pix_model(data, mode='generator',
                                                                                  iter_count=iter_count)
        if g_loss_for_backward is not None:
            g_loss_for_backward.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.pix2pix_model_on_one_gpu.netG.parameters(), 1.0)
            self.optimizer_G.step()
        self.g_losses = g_losses_for_display
        self.generated = generated

    def run_discriminator_one_step(self, data, iter_count):
        if self.optimizer_D is None:
            self.d_losses = {}
            return

        self.optimizer_D.zero_grad()
        _, d2_losses = self.pix2pix_model(data, mode='discriminator', iter_count=iter_count)

        d_loss = sum(d2_losses.values()).mean()

        d_loss.backward()
        self.optimizer_D.step()

        self.d_losses = d2_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            if self.opt.niter_decay > 0:
                lrd = self.opt.lr / self.opt.niter_decay
                new_lr = self.old_lr - lrd
            else:
                new_lr = self.old_lr
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr
                new_lr_D = new_lr / 2

            print(f'Updating learning rate: old_base={self.old_lr:.6f} -> new_base={new_lr:.6f}')
            print(f'New LR_G: {new_lr_G:.6f}, New LR_D: {new_lr_D:.6f}')

            if self.optimizer_D:
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] = new_lr_D

            if self.optimizer_G:
                for param_group in self.optimizer_G.param_groups:
                    param_group['lr'] = new_lr_G

            self.old_lr = new_lr