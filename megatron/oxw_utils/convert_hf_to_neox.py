import torch
from megatron import print_rank_0

from transformers.models.gpt2 import GPT2LMHeadModel
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.model.transformer import (
    ParallelTransformerLayerPipe,
    NormPipe,
    ParallelLinearPipe,
)
from megatron.model.gpt2_model import GPT2ModelPipe
from megatron.model.word_embeddings import EmbeddingPipe
from types import FunctionType

# nested getattr + setattr
# from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
import functools


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def convert_hf_to_neox(hf_model: GPT2LMHeadModel, model_pipe: GPT2ModelPipe, args: NeoXArgs):

    assert type(model_pipe) == GPT2ModelPipe

    # mapping for layers
    layer_idx_transformer_layers = []
    layer_idx_embedding_layers = []
    layer_functions = []
    layer_idx_norm_pipe_layers = []
    layer_idx_linear_pipe_layers = []

    errors = []

    for idx, ff in enumerate(model_pipe.forward_funcs):

        if isinstance(ff, ParallelTransformerLayerPipe):
            layer_idx_transformer_layers.append(
                (idx, ff)
            )
        elif isinstance(ff, EmbeddingPipe):
            layer_idx_embedding_layers.append(
                (idx, ff)
            )
        elif isinstance(ff, FunctionType):  # callable(ff):
            layer_functions.append(
                (idx, ff)
            )
        elif isinstance(ff, NormPipe):
            layer_idx_norm_pipe_layers.append(
                (idx, ff)
            )
        elif isinstance(ff, ParallelLinearPipe):
            layer_idx_linear_pipe_layers.append(
                (idx, ff)
            )
        else:
            errors += ('Not matched: ', type(ff))

    if errors:
        raise ValueError(errors)

    assert len(layer_idx_transformer_layers) == args.num_layers
    assert len(hf_model.transformer.h) == args.num_layers
    assert len(layer_idx_embedding_layers) == 1
    assert len(layer_idx_norm_pipe_layers) == 1
    assert len(layer_idx_linear_pipe_layers) == 1

    # change weights
    with torch.no_grad():
        # word embedding
        source_layer = hf_model.transformer.wte
        print_rank_0(type(source_layer))

        source_state_dict = source_layer.state_dict()
        target_idx, _ = layer_idx_embedding_layers[0]

        for source_k in source_state_dict.keys():
            print_rank_0(' - ', source_k)
            rsetattr(
                model_pipe.forward_funcs[target_idx].word_embeddings,
                source_k,
                rgetattr(source_layer, source_k)
            )

        # position embedding
        source_layer = hf_model.transformer.wpe
        print_rank_0(type(source_layer))

        source_state_dict = source_layer.state_dict()
        target_idx, _ = layer_idx_embedding_layers[0]

        for source_k in source_state_dict.keys():
            print_rank_0(' - ', source_k)
            rsetattr(
                model_pipe.forward_funcs[target_idx].position_embeddings,
                source_k,
                rgetattr(source_layer, source_k)
            )

        # transformer layers
        key_mapping = {
            # layer norm 2
            'ln_1.weight': 'input_layernorm.weight',
            'ln_1.bias': 'input_layernorm.bias',
            # attention
            'attn.c_attn.weight': 'attention.query_key_value.weight',
            'attn.c_attn.bias': 'attention.query_key_value.bias',
            'attn.c_proj.weight': 'attention.dense.weight',
            'attn.c_proj.bias': 'attention.dense.bias',
            # layer norm 2
            'ln_2.weight': 'post_attention_layernorm.weight',
            'ln_2.bias': 'post_attention_layernorm.bias',
            # mlp
            'mlp.c_fc.weight': 'mlp.dense_h_to_4h.weight',
            'mlp.c_fc.bias': 'mlp.dense_h_to_4h.bias',
            'mlp.c_proj.weight': 'mlp.dense_4h_to_h.weight',
            'mlp.c_proj.bias': 'mlp.dense_4h_to_h.bias',
            # ignore --> all equal
            # 'attn.bias', 'attn.masked_bias'
        }
        for i in range(args.num_layers):
            source_layer = hf_model.transformer.h[i]
            print_rank_0(type(source_layer))

            source_state_dict = source_layer.state_dict()

            target_idx, _ = layer_idx_transformer_layers[i]

            for hf_k, neox_k in key_mapping.items():
                print(' - ', hf_k)
                rsetattr(
                    model_pipe.forward_funcs[target_idx],
                    neox_k,
                    rgetattr(source_layer, hf_k)
                )

                # layer norm
        source_layer = hf_model.transformer.ln_f
        print_rank_0(type(source_layer))

        source_state_dict = source_layer.state_dict()
        target_idx, _ = layer_idx_norm_pipe_layers[0]

        for source_k in source_state_dict.keys():
            print_rank_0(' - ', source_k)
            rsetattr(
                model_pipe.forward_funcs[target_idx],
                source_k,
                rgetattr(source_layer, source_k)
            )

        # lm head
        source_layer = hf_model.lm_head
        print_rank_0(type(source_layer))

        source_state_dict = source_layer.state_dict()
        target_idx, _ = layer_idx_linear_pipe_layers[0]

        for source_k in source_state_dict.keys():
            print_rank_0(' - ', source_k)
            rsetattr(
                model_pipe.forward_funcs[target_idx],
                source_k,
                rgetattr(source_layer, source_k)
            )

    return model_pipe
