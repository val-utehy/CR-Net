import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.encoder import ConvEncoder
from models.networks.generator import TSITGenerator, Pix2PixHDGenerator
from models.networks.rafael_generator import RafaelGenerator
import torch.nn as nn
from .style_encoder import SimpleStyleEncoder

SWIN_COND_D_IMPORTED = False
SwinTransformerConditionalDiscriminator = None
try:
    from models.networks.swin_discriminator import SwinTransformerConditionalDiscriminator
    SWIN_COND_D_IMPORTED = True
    print("Successfully imported SwinTransformerConditionalDiscriminator.")
except ImportError as e:
    print(f"Warning: Could not import SwinTransformerConditionalDiscriminator: {e}. Swin D will not be available.")
except Exception as e:
    print(f"Warning: An unexpected error occurred while importing SwinTransformerConditionalDiscriminator: {e}")


def find_class_in_module(target_cls_name, module_globals):
    target_cls_name_lower_no_underscore = target_cls_name.lower().replace('_', '')
    for name, cls_obj in module_globals.items():
        if isinstance(cls_obj, type) and name.lower().replace('_', '') == target_cls_name_lower_no_underscore:
            return cls_obj

    available_classes = [name for name, obj in module_globals.items() if isinstance(obj, type)]
    print(
        f"Error: Class '{target_cls_name_lower_no_underscore}' (derived from '{target_cls_name}') not found in module_globals.")
    print(f"Available classes (case-insensitive, no underscore): {[ac.lower().replace('_', '') for ac in available_classes]}")
    print(f"Original available classes: {available_classes}")
    raise ValueError(
        f"In current module, there should be a class named '{target_cls_name}' (comparison is case-insensitive and ignores underscores)."
    )


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    # Generator options
    if hasattr(opt, 'netG') and opt.netG:
        try:
            netG_cls = find_class_in_module(opt.netG, globals())
            if hasattr(netG_cls, 'modify_commandline_options'):
                parser = netG_cls.modify_commandline_options(parser, is_train)
                print(f"DEBUG: Called modify_commandline_options for netG: {opt.netG}")
        except ValueError as e:
            print(f"Warning: Could not find class for netG '{opt.netG}' to call modify_commandline_options. {e}")


    # Discriminator options
    if is_train and hasattr(opt, 'netD') and opt.netD:
        netD_cls_to_modify = None
        netD_lower_no_underscore = opt.netD.lower().replace('_','')

        if netD_lower_no_underscore == 'swintransformerconditionaldiscriminator' and SWIN_COND_D_IMPORTED:
            netD_cls_to_modify = SwinTransformerConditionalDiscriminator
            print(f"DEBUG: networks.modify_commandline_options trying SwinTransformerConditionalDiscriminator for netD '{opt.netD}'.")
        elif netD_lower_no_underscore.startswith('swin') and SWIN_COND_D_IMPORTED:
            netD_cls_to_modify = SwinTransformerConditionalDiscriminator
            print(f"DEBUG: networks.modify_commandline_options trying SwinTransformerConditionalDiscriminator for netD '{opt.netD}' by prefix.")
        else:
            try:
                cls_cand = find_class_in_module(opt.netD, globals())
                if issubclass(cls_cand, BaseNetwork):
                    netD_cls_to_modify = cls_cand
                    print(f"DEBUG: networks.modify_commandline_options trying {cls_cand.__name__} for netD '{opt.netD}'.")
            except ValueError:
                 pass

        if netD_cls_to_modify and hasattr(netD_cls_to_modify, 'modify_commandline_options'):
            parser = netD_cls_to_modify.modify_commandline_options(parser, is_train)
            print(f"DEBUG: Called modify_commandline_options for netD: {netD_cls_to_modify.__name__}")
        elif netD_lower_no_underscore.startswith('swin') and not netD_cls_to_modify:
            print(f"Warning: netD '{opt.netD}' looks like Swin, but class/modify_commandline_options not found/called.")
        elif not netD_cls_to_modify:
            print(f"Warning: Could not find a specific class for netD '{opt.netD}' to call modify_commandline_options. Standard D options might not be fully parsed if this D type has specific ones.")


    # VAE Encoder options (original TSIT)
    if hasattr(opt, 'use_vae') and opt.use_vae:
        if not hasattr(opt, 'netG') or opt.netG != 'RafaelGenerator': # VAE not used with RafaelGenerator in current setup
            if hasattr(ConvEncoder, 'modify_commandline_options'):
                parser = ConvEncoder.modify_commandline_options(parser, is_train)
                print("DEBUG: Called modify_commandline_options for ConvEncoder (netE).")
    return parser


def define_G(opt):
    # opt.netG should be the string name of the generator class (e.g., "RafaelGenerator")
    # find_class_in_module will find it in the `globals()` of this file.
    print(f"Attempting to define Generator: {opt.netG}")
    netG_cls = find_class_in_module(opt.netG, globals())
    netG = netG_cls(opt)

    print_network(netG)
    if len(opt.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(opt.gpu_ids[0]) # .cuda() is deprecated, use .to(device)
        # netG.to(torch.device(f'cuda:{opt.gpu_ids[0]}'))
    netG.init_weights(opt.init_type, opt.init_variance)
    return netG


def define_D(opt):
    netD_cls = None
    netD_name_lower_no_underscore = opt.netD.lower().replace('_', '')

    print(f"Attempting to define Discriminator for opt.netD = '{opt.netD}'")

    if netD_name_lower_no_underscore == 'swintransformerconditionaldiscriminator' and SWIN_COND_D_IMPORTED:
        print(f"Selected SwinTransformerConditionalDiscriminator for netD by full name: '{opt.netD}'.")
        netD_cls = SwinTransformerConditionalDiscriminator
    elif netD_name_lower_no_underscore.startswith("swin") and SWIN_COND_D_IMPORTED:
        print(f"Selected SwinTransformerConditionalDiscriminator for netD by prefix: '{opt.netD}'.")
        netD_cls = SwinTransformerConditionalDiscriminator
    elif opt.netD in ['MultiscaleDiscriminator', 'NLayerDiscriminator']: # Add other known D types here
        print(f"Selected {opt.netD} for netD (classic type).")
        netD_cls = find_class_in_module(opt.netD, globals())
    else:
        # Fallback for any other D specified, or if the above specific checks failed
        try:
            print(f"Attempting to find class for netD '{opt.netD}' by general search in globals()...")
            netD_cls = find_class_in_module(opt.netD, globals()) # This will raise ValueError if not found
            print(f"Found and selected {netD_cls.__name__} for netD via general search.")
        except ValueError as e:
            # This will now properly raise an error if opt.netD is not recognized.
             raise ValueError(f"Unknown or unimported Discriminator type specified by --netD: {opt.netD}. Error: {e}")


    if netD_cls is None: # Should ideally not be reached if find_class_in_module or the specific checks work
        raise ValueError(f"Could not assign a class for Discriminator type: {opt.netD}. Check imports and naming.")

    netD = netD_cls(opt)
    print_network(netD)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        netD.cuda(opt.gpu_ids[0])
        # netD.to(torch.device(f'cuda:{opt.gpu_ids[0]}'))

    netD.init_weights(opt.init_type, opt.init_variance)
    return netD


def define_E(opt):
    print("Attempting to define Encoder (ConvEncoder for VAE)")
    netE = ConvEncoder(opt)
    print_network(netE)
    if len(opt.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netE.cuda(opt.gpu_ids[0])
        # netE.to(torch.device(f'cuda:{opt.gpu_ids[0]}'))
    netE.init_weights(opt.init_type, opt.init_variance)
    return netE


def print_network(net):
    if isinstance(net, list):
        actual_net_to_print = net[0]
        print(f"Printing network info for the first model in a list of {len(net)} models.")
    else:
        actual_net_to_print = net

    num_params = 0
    for param in actual_net_to_print.parameters():
        num_params += param.numel()

    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).'
          % (type(actual_net_to_print).__name__, num_params / 1000000))


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(opt, norm_type='instance'):
    # norm_type is a string like 'spectralinstance' or 'instance' or 'batch'
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        if hasattr(layer, 'out_features'): # For Linear layers
            return getattr(layer, 'out_features')
        if hasattr(layer, 'weight') and layer.weight.dim() > 1: # Conv, Linear
            return layer.weight.size(0)
        raise ValueError(f"Cannot get out_channel for layer {type(layer)}")

    def add_norm_layer(layer_instance): # layer_instance is e.g. nn.Conv2d(...)
        nonlocal norm_type # Use norm_type from the outer scope
        current_norm_type = norm_type

        # Spectral normalization part
        if current_norm_type.startswith('spectral'):
            # Apply spectral norm to the layer (e.g., nn.Conv2d)
            # torch.nn.utils.spectral_norm returns the spectrally_normed_layer
            layer_with_spec_norm = torch.nn.utils.spectral_norm(layer_instance)
            # The rest of the norm_type string is the actual normalization (e.g., 'instance')
            subnorm_type = current_norm_type[len('spectral'):]
        else:
            layer_with_spec_norm = layer_instance # No spectral norm
            subnorm_type = current_norm_type

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer_with_spec_norm # Only spectral norm (if any) or just the layer

        # For other normalizations, remove bias from the layer_with_spec_norm
        # as it's cancelled by the normalization layer's learnable/non-learnable beta.
        if getattr(layer_with_spec_norm, 'bias', None) is not None:
            delattr(layer_with_spec_norm, 'bias')
            layer_with_spec_norm.register_parameter('bias', None)

        out_channels = get_out_channel(layer_instance) # Get channels from original layer

        if subnorm_type == 'batch':
            norm_layer_instance = nn.BatchNorm2d(out_channels, affine=True)
        elif subnorm_type == 'syncbatch':
            # Assuming SynchronizedBatchNorm2d is available in this scope
            # from models.networks.sync_batchnorm import SynchronizedBatchNorm2d # Or import globally
            try:
                from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
                norm_layer_instance = SynchronizedBatchNorm2d(out_channels, affine=True)
            except ImportError:
                print("Warning: SynchronizedBatchNorm2d not found, falling back to BatchNorm2d for 'syncbatch'.")
                norm_layer_instance = nn.BatchNorm2d(out_channels, affine=True)
        elif subnorm_type == 'instance':
            norm_layer_instance = nn.InstanceNorm2d(out_channels, affine=False) # Original TSIT used affine=False
        elif subnorm_type.startswith('fade'): # FADE is handled differently, usually within the block
            print(
                f"Warning: Norm type '{subnorm_type}' looks like FADE. "
                "This generic get_norm_layer is not for FADE. FADE should be part of specific blocks. "
                "Returning layer without additional FADE-like normalization here.")
            return layer_with_spec_norm # Return layer possibly with spectral norm
        else:
            raise ValueError('normalization layer %s (from %s) is not recognized' % (subnorm_type, norm_type))

        return nn.Sequential(layer_with_spec_norm, norm_layer_instance)

    return add_norm_layer