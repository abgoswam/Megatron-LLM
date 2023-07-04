import torch
from transformers import AutoModelForCausalLM

import megatron
from megatron import get_args, print_rank_0, update_num_microbatches
from megatron.model.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.training import _setup_model_and_optimizer, build_train_valid_test_data_iterators

from finetune_falcon import (add_args, _get_batch, loss_func, _model_provider,
                             _train_valid_test_datasets_provider)


def falcon_provider(size: int = 7):
    args = get_args()
    print_rank_0("Getting falcon model...")
    model = AutoModelForCausalLM.from_pretrained(
        f"tiiuae/falcon-{size}b", cache_dir=args.huggingface_cache,
        trust_remote_code=True
    )
    model = model.eval().requires_grad_(False).to(args.huggingface_device)
    return model



def megatron_forward(model, batch):
    model = model[0]  # TODO: asuming no model parallelism
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    assert torch.all(loss_mask)
    # we need to do two forward passes to get both the logits and the loss
    logits = model(tokens, position_ids, attention_mask, labels=None)
    losses = model(tokens, position_ids, attention_mask, labels=labels)
    loss, _ = loss_func(loss_mask, losses)
    return logits, loss


def huggingface_forward(model, batch):
    args = get_args()
    batch = [tensor.to(args.huggingface_device) for tensor in batch]
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    output = model(input_ids=tokens, position_ids=position_ids, labels=tokens,
                   output_hidden_states=True)
    return output["logits"], output["loss"]


def verify_step(forward1, model1, forward2, model2, iterator):
    batch = _get_batch(iterator)
    logits1, loss1 = forward1(model1, batch)
    logits2, loss2 = forward2(model2, batch)
    assert logits1.size() == logits2.size()
    assert loss1.size() == loss2.size()
    logits1 = logits1.cpu()
    logits2 = logits2.cpu()
    loss1 = loss1.cpu()
    loss2 = loss2.cpu()
    abs_error = torch.max(torch.abs(logits1 - logits2))
    loss_error = torch.abs(loss1 - loss2)
    print("Max absoulute error in the logits: "
          f"{abs_error:.3f} Abs loss error: {loss_error:.3f}")


# heavily inspired from megatron.training.pretrain
def verify(args, train_valid_test_dataset_provider,
           model_provider_func, forward_step_func,
           baseline_provider_func, forward_baseline_func,
           model_type, process_non_loss_data_func=None):
    """Verifies that the `forward_step_func` gives forward outputs similar
    to the `forward_baseline_func` using the dataset provider specified"""

    # MISC INITALIZATIONS
    megatron.initialize.set_jit_fusion_options(args)
    print_rank_0("Megatron has been initialized!")
    model, optimizer, opt_param_schedulkr = _setup_model_and_optimizer(
        model_provider_func, model_type, args=args
    )
    for module in model:
        module.eval().requires_grad_(False)
    baseline_model = baseline_provider_func()
    print_rank_0("Model has been setup")
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider, args)
            for _ in range(len(model))
        ]
        train_data_iterator = [di[0] for di in all_data_iterators]
        valid_data_iterator = [di[1] for di in all_data_iterators]
        test_data_iterator = [di[2] for di in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider, args)
    print_rank_0("Dataloaders have been built!")

    # Now we can start the verifications
    print_rank_0("Starting verifications!")
    for iteration in range(args.iteration, args.train_iters):
        print_rank_0(f"Iteration {iteration}...")
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        verify_step(forward_step_func, model,
                    forward_baseline_func, baseline_model, train_data_iterator)


def extra_args(parser):
    parser = add_args(parser)
    group = parser.add_argument_group(title="huggingface")
    group.add_argument("--huggingface_cache", default=None)
    group.add_argument("--huggingface_device", default="cuda:0")
    return parser


if __name__ == "__main__":
    # INITIALIZATION
    print("Starting falcon-megatron vs falcon-huggingface verification")
    initialize_megatron(extra_args, {"tokenizer_type": "FalconTokenizer"})
    args = get_args()

    # VERIFICATION
    verify(args, _train_valid_test_datasets_provider,
           _model_provider, megatron_forward,
           falcon_provider, huggingface_forward,
           ModelType.encoder_or_decoder,)
    print(model)
    print("Verification done, we are set now woohoo! :)")
