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

    hf_vocab_size = len(hf_model.transformer.wte.weight)

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
        target_idx, _ = layer_idx_embedding_layers[0]
        # - handle vocab mismatch -> only override until vocab_size is reached
        model_pipe.forward_funcs[target_idx].word_embeddings.weight[:hf_vocab_size, :] = source_layer.weight[:, :]

        # position embedding
        source_layer = hf_model.transformer.wpe
        print_rank_0(type(source_layer))
        target_idx, _ = layer_idx_embedding_layers[0]
        model_pipe.forward_funcs[target_idx].position_embeddings.weight[:, :] = source_layer.weight[:, :]

        # transformer blocks
        for i in range(args.num_layers):
            source_layer = hf_model.transformer.h[i]
            print_rank_0(type(source_layer))

            target_idx, _ = layer_idx_transformer_layers[i]

            # layer norm
            model_pipe.forward_funcs[target_idx].input_layernorm.weight[:] = source_layer.ln_1.weight[:]
            model_pipe.forward_funcs[target_idx].input_layernorm.bias[:] = source_layer.ln_1.bias[:]

            # attention
            model_pipe.forward_funcs[target_idx].attention.query_key_value.weight[:] = source_layer.attn.c_attn.weight.T[:]
            model_pipe.forward_funcs[target_idx].attention.query_key_value.bias[:] = source_layer.attn.c_attn.bias[:]
            model_pipe.forward_funcs[target_idx].attention.dense.weight[:] = source_layer.attn.c_proj.weight[:]
            model_pipe.forward_funcs[target_idx].attention.dense.bias[:] = source_layer.attn.c_proj.bias[:]

            # layer norm 2
            model_pipe.forward_funcs[target_idx].post_attention_layernorm.weight[:] = source_layer.ln_2.weight[:]
            model_pipe.forward_funcs[target_idx].post_attention_layernorm.bias[:] = source_layer.ln_2.bias[:]

            # mlp
            model_pipe.forward_funcs[target_idx].mlp.dense_h_to_4h.weight[:] = source_layer.mlp.c_fc.weight.T[:]
            model_pipe.forward_funcs[target_idx].mlp.dense_h_to_4h.bias[:] = source_layer.mlp.c_fc.bias[:]

            model_pipe.forward_funcs[target_idx].mlp.dense_4h_to_h.weight[:] = source_layer.mlp.c_proj.weight.T[:]
            model_pipe.forward_funcs[target_idx].mlp.dense_4h_to_h.bias[:] = source_layer.mlp.c_proj.bias[:]

            # ignore --> all equal
            # 'attn.bias', 'attn.masked_bias'
    
        # norm
        source_layer = hf_model.transformer.ln_f
        print_rank_0(type(source_layer))

        target_idx, _ = layer_idx_norm_pipe_layers[0]
        model_pipe.forward_funcs[target_idx].norm.weight[:] = source_layer.weight[:]

        # lm head
        source_layer = hf_model.lm_head
        print_rank_0(type(source_layer))

        target_idx, _ = layer_idx_linear_pipe_layers[0]
        model_pipe.forward_funcs[target_idx].final_linear.weight[:hf_vocab_size,:] = source_layer.weight[:]


    return model_pipe
