def parse_model(model):
    if "opt" in str(type(model)).lower():
        model_type = "opt"
    else:
        # TODO: for now, we assume that if it's not opt, it's llama variant
        # additional rules should be added to support other models
        model_type = "llama"
    print(f'Model type : {model_type}')

    return model_type

def get_module_names(model_type):
    if model_type == "opt":
        return ["q", "k", "v", "o", "up", "down"]
    else:
        assert model_type == "llama"
        return ["q", "k", "v", "o", "gate", "up", "down"]


def get_sequential(model_type):
    if model_type == "opt":
        return [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.out_proj',
            'fc1',
            'fc2',
        ]
    else:
        assert model_type == "llama"
        return [
            'self_attn.q_proj', 
            'self_attn.k_proj', 
            'self_attn.v_proj',
            'self_attn.o_proj',
            'mlp.gate_proj',
            'mlp.up_proj', 
            'mlp.down_proj',
        ]


def get_model(model, model_type):
    if model_type == 'opt':
        return model.model.decoder
    else:
        assert model_type == 'llama'
        return model.model

def get_layers(model, model_type):
    _model = get_model(model, model_type)
    if model_type == 'opt':
        return _model.layers
    else:
        assert model_type == 'llama'
        return _model.layers

def get_layers_name(model_type):
    if model_type == 'opt':
        return 'model.decoder.layers'
    else:
        assert model_type == 'llama'
        return 'model.layers'

def get_embedding(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return [_model.embed_tokens, _model.embed_positions]
    else:
        assert model_type == "llama"
        return [_model.embed_tokens]

def get_norm(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.final_layer_norm
    else:
        assert model_type == "llama"
        return _model.norm
