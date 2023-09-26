import os
import re
import sys

from modules import sd_models, shared
from modules.paths import extensions_builtin_dir
from scripts.animatediff_logger import logger_animatediff as logger

sys.path.append(f"{extensions_builtin_dir}/Lora")

class AnimateDiffLora:

    def __init__(self, v2: bool):
        self.original_load_network = None
        self.original_convert = None
        self.v2 = v2

    def hack(self):
        if not self.v2:
            return

        logger.info("Hacking lora to support motion lora")
        import network, networks
        self.original_load_network = networks.load_network
        self.original_convert = networks.convert_diffusers_name_to_compvis
        original_load_network = self.original_load_network
        original_convert = self.original_convert

        def mm_load_network(name, network_on_disk):

            def convert_mm_name_to_compvis(key):
                sd_module_key, _, network_part = re.split(r'(_lora\.)', key)
                sd_module_key = sd_module_key.replace('processor.', '')
                if 'mid_block' in sd_module_key:
                    match = re.match(r'mid_block\.motion_modules\.(\d+)(.*)', sd_module_key)
                    sd_module_key = f'diffusion_model_middle_block_2{match.group(2)}'
                elif 'down_blocks' in sd_module_key:
                    match = re.match(r'down_blocks\.(\d+)\.motion_modules\.(\d+)(.*)', sd_module_key)
                    b = int(match.group(1))
                    m = int(match.group(2))
                    idx = [1, 2, 4, 5, 7, 8, 10, 11][b * 2 + m]
                    sd_module_key = f'diffusion_model_input_blocks_{idx}_3{match.group(3)}'
                elif 'up_blocks' in sd_module_key:
                    match = re.match(r'up_blocks\.(\d+)\.motion_modules\.(\d+)(.*)', sd_module_key)
                    b = int(match.group(1))
                    m = int(match.group(2))
                    sd_module_key = f'diffusion_model_output_blocks_{b * 3 + m}_2{match.group(3)}'

                sd_module_key = sd_module_key.replace('.', '_')
                return sd_module_key, 'lora_' + network_part

            net = network.Network(name, network_on_disk)
            net.mtime = os.path.getmtime(network_on_disk.filename)

            sd = sd_models.read_state_dict(network_on_disk.filename)

            if not hasattr(shared.sd_model, 'network_layer_mapping'):
                networks.assign_network_names_to_compvis_modules(shared.sd_model)
            
            if 'motion_modules' in list(sd.keys())[0]:
                logger.info(f"Loading motion lora {name} from {network_on_disk.filename}")
                matched_networks = {}

                for key_network, weight in sd.items():
                    key, network_part = convert_mm_name_to_compvis(key_network)
                    sd_module = shared.sd_model.network_layer_mapping.get(key, None)

                    assert sd_module is not None, f"Failed to find sd module for key {key}."

                    if key not in matched_networks:
                        matched_networks[key] = network.NetworkWeights(
                            network_key=key_network, sd_key=key, w={}, sd_module=sd_module)

                    matched_networks[key].w[network_part] = weight

                for key, weights in matched_networks.items():
                    net_module = networks.module_types[0].create_module(net, weights)
                    assert net_module is not None, "Failed to create motion module lora"
                    net.modules[key] = net_module

                return net
            else:
                del sd
                return original_load_network(name, network_on_disk)
        
        def mm_convert(key, is_sd2):
            conversion: str = original_convert(key, is_sd2)
            if conversion[:30] == 'diffusion_model_middle_block_2':
                return conversion.replace('diffusion_model_middle_block_2', 'diffusion_model_middle_block_3')
            if conversion[:30] == 'diffusion_model_output_blocks_' and conversion[30] in ['2', '5', '8'] and conversion[31:] == '_2_conv':
                return conversion.replace('_2_conv', '_3_conv')
            return conversion

        networks.load_network = mm_load_network
        networks.convert_diffusers_name_to_compvis = mm_convert

    
    def restore(self):
        if self.v2:
            logger.info("Restoring hacked lora")
            import networks
            networks.load_network = self.original_load_network
            networks.convert_diffusers_name_to_compvis = self.original_convert
