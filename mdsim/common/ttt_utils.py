
def gemnet_frozen_params(name):
    if 'out_blocks.4' not in name and 'int_blocks.3' not in name:
        return True
    else:
        return False


def gemnet_frozen_params_med(name):
    if 'out_blocks.4' not in name and 'int_blocks.3' not in name and 'out_blocks.3' not in name and 'int_blocks.2' not in name:
        return True
    else:
        return False

def gemnet_frozen_params_min(name):
    if 'atom_emb' in name or 'edge_emb' in name or 'out_blocks.0' in name or 'int_blocks.0' in name:
        return True
    else:
        return False
    
gemnet_freeze_amount_to_fn = {
    'min': gemnet_frozen_params_min,
    'med': gemnet_frozen_params_med,
    'max': gemnet_frozen_params,
}