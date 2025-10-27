import json
import time
import math
import numpy as np
import transformers
import mlx.nn as nn
import mlx.core as mx
from pathlib import Path
import mlx.optimizers as optim
from pydantic import BaseModel
from typing import Tuple, Generator, Optional, Union
from mlx.utils import tree_flatten, tree_unflatten

from .models import Model, LoRALinear
from .utils import load, generate, save_model, upload_to_hub

class Dataset:
    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)
    
class LORAConfig(BaseModel):
    train: bool = True
    lora_layers: int = 16
    prompt: str | None = None
    temp: float = 0.8
    max_tokens: int = 100
    learning_rate: float = 1e-5
    iterations: int = 1000
    val_batches: int = 25
    batch_size: int = 4
    steps_per_report: int = 10
    steps_per_eval: int = 200
    resume_adapter_file: str | None = None
    adapter_file: str = "adapters.npz"
    save_every: int = 100
    test: bool = False
    test_batches: int = 500
    seed: int = 0

class FUSEConfig(BaseModel):
    adapter_file: str = "adapters.npz"
    hf_path: str | None = None
    upload_name: str | None = None
    de_quantize: bool = False

class LORA:
    def __init__(self, config: Optional[Union[LORAConfig, dict]] = None) -> None:
        if config is None:
            cfg = LORAConfig()
        elif isinstance(config, LORAConfig):
            cfg = config
        elif isinstance(config, dict):
            try:
                cfg = LORAConfig.model_validate(config)
            except Exception:
                cfg = LORAConfig(**config)
        else:
            try:
                data = dict(config)
            except Exception:
                data = {}
            try:
                cfg = LORAConfig.model_validate(data)
            except Exception:
                cfg = LORAConfig(**data)

        self.config = cfg

    def _load_datasets(self, data: str):
        def load_and_check(name):
            dataset_path = Path(data) / f"{name}.jsonl"
            try:
                return Dataset(dataset_path)
            except Exception as e:
                print(f"Unable to build dataset {dataset_path} ({e})")
                raise

        names = ("train", "valid", "test")
        train, valid, test = (load_and_check(n) for n in names)
        return train, valid, test

    def _loss(self, model: Model, inputs: mx.array, targets: mx.array, lengths: mx.array) -> Tuple[mx.array, int]:
        logits, _ = model(inputs)
        logits = logits.astype(mx.float32)
        length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
        ce = nn.losses.cross_entropy(logits, targets) * length_mask
        ntoks = length_mask.sum()
        ce = ce.sum() / ntoks
        return ce, ntoks
    
    def _iterate_batches(self, dataset: Dataset, tokenizer: transformers.PreTrainedTokenizer, batch_size: int, train: bool = False) -> Generator[Tuple[mx.array, mx.array, mx.array], None, None]:
        while True:
            indices = np.arange(len(dataset))
            if train:
                indices = np.random.permutation(indices)
            for i in range(0, len(indices) - batch_size + 1, batch_size):
                batch = [tokenizer.encode(dataset[indices[i + j]]) for j in range(batch_size)]
                lengths = [len(x) for x in batch]
                if max(lengths) > 2048:
                    print("[WARNING] Some sequences are longer than 2048 tokens. Consider pre-splitting your data to save memory.")
                batch_arr = np.zeros((batch_size, max(lengths)), np.int32)
                for j in range(batch_size):
                    batch_arr[j, : lengths[j]] = batch[j]
                batch = mx.array(batch_arr)
                yield batch[:, :-1], batch[:, 1:], mx.array(lengths)
            if not train:
                break

    def _evaluate(self, model: Model, dataset: Dataset, tokenizer: transformers.PreTrainedTokenizer, batch_size: int, num_batches: int) -> float:
        all_losses = []
        ntokens = 0
        index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
        for it, batch in zip(index_iterator, self._iterate_batches(dataset, tokenizer, batch_size)):
            losses, toks = self._loss(model, *batch)
            all_losses.append((losses * toks).item())
            ntokens += toks.item()
        return np.sum(all_losses) / ntokens

    def _train(self, model: Model, train_set: Dataset, valid_set: Dataset, optimizer: optim.Optimizer, tokenizer: transformers.PreTrainedTokenizer, iterations: int, batch_size: int, steps_per_report: int, steps_per_eval: int, val_batches: int, save_every: int, adapter_file: str) -> None:
        loss_value_and_grad = nn.value_and_grad(model, self._loss)
        losses = []
        n_tokens = 0

        start = time.perf_counter()
        for it, batch in zip(range(iterations), self._iterate_batches(train_set, tokenizer, batch_size, train=True)):
            (lvalue, toks), grad = loss_value_and_grad(model, *batch)
            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state, lvalue)
            losses.append(lvalue.item())
            n_tokens += toks.item()

            if (it + 1) % steps_per_report == 0:
                train_loss = np.mean(losses)
                stop = time.perf_counter()
                print(
                    f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                    f"It/sec {steps_per_report / (stop - start):.3f}, "
                    f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
                )
                losses = []
                n_tokens = 0
                start = time.perf_counter()

            if it == 0 or (it + 1) % steps_per_eval == 0:
                stop = time.perf_counter()
                val_loss = self._evaluate(model, valid_set, tokenizer, batch_size, val_batches)
                print(
                    f"Iter {it + 1}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {(time.perf_counter() - stop):.3f}s"
                )
                start = time.perf_counter()

            if (it + 1) % save_every == 0:
                mx.savez(adapter_file, **dict(tree_flatten(model.trainable_parameters())))
                print(f"Iter {it + 1}: Saved adapter weights to {adapter_file}.")

    def _generate(self, model: Model, prompt: str, tokenizer: transformers.PreTrainedTokenizer, temp: float, max_tokens: int):
        print(prompt, end="", flush=True)
        prompt = mx.array(tokenizer.encode(prompt))
        tokens = []
        skip = 0
        for token, n in zip(generate(prompt, model, temp), range(max_tokens)):
            if token == tokenizer.eos_token_id:
                break
            tokens.append(token.item())
            s = tokenizer.decode(tokens)
            if len(s) - skip > 1:
                print(s[skip:-1], end="", flush=True)
                skip = len(s) - 1
        print(tokenizer.decode(tokens)[skip:], flush=True)
        print("=" * 10)
        if len(tokens) == 0:
            print("No tokens generated for this prompt")
            return
        
    def invoke(self, model_path: str, data: str = "data/") -> None:
        np.random.seed(self.config.seed)

        tokenizer_config = {}
        if self.config.train: 
            tokenizer_config["add_eos_token"] = True

        model, tokenizer, _ = load(model_path, tokenizer_config)
        model.freeze()
        for l in model.model.layers[len(model.model.layers) - self.config.lora_layers:]:
            l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

        p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
        print(f"Total parameters {p:.3f}M")
        p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
        print(f"Trainable parameters {p:.3f}M")

        print("Loading datasets")
        train_set, valid_set, test_set = self._load_datasets(data)

        if self.config.resume_adapter_file is not None:
            print(f"Loading pretrained adapters from {self.config.resume_adapter_file}")
            model.load_weights(self.config.resume_adapter_file, strict=False)

        if self.config.train:
            print("Training")
            opt = optim.Adam(learning_rate=self.config.learning_rate)
            self._train(model, train_set, valid_set, opt, tokenizer, self.config.iterations, self.config.batch_size, self.config.steps_per_report, self.config.steps_per_eval, self.config.val_batches, self.config.save_every, self.config.adapter_file)
            mx.savez(self.config.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

        if not Path(self.config.adapter_file).is_file():
            raise ValueError(f"Adapter file {self.config.adapter_file} missing! Use 'train=True' to learn and save the adapters.")

        model.load_weights(self.config.adapter_file, strict=False)

        if self.config.test:
            print('Testing')
            model.eval()
            test_loss = self._evaluate(model=model, dataset=test_set, tokenizer=tokenizer, batch_size=self.config.batch_size, num_batches=self.config.val_batches)
            test_ppl = math.exp(test_loss)
            print(f"Test loss {test_loss:.3f}, Test PPL {test_ppl:.3f}")

        if self.config.prompt is not None:
            print("Generating")
            self._generate(model, self.config.prompt, tokenizer, self.config.temp, self.config.max_tokens)

class FUSE:
    def __init__(self, config: Optional[Union[FUSEConfig, dict]] = None) -> None:
        if config is None:
            cfg = FUSEConfig()
        elif isinstance(config, FUSEConfig):
            cfg = config
        elif isinstance(config, dict):
            try:
                cfg = FUSEConfig.model_validate(config)
            except Exception:
                cfg = FUSEConfig(**config)
        else:
            try:
                data = dict(config)
            except Exception:
                data = {}
            try:
                cfg = FUSEConfig.model_validate(data)
            except Exception:
                cfg = FUSEConfig(**data)

        self.config = cfg

    def invoke(self, model_path: str, save_path: str = "lora_fused_model") -> None:
        model, tokenizer, model_config = load(model_path)

        adapters = list(mx.load(self.config.adapter_file).items())
        lora_layers = len([m for m in adapters if "q_proj.lora_a" in m[0]])

        model.freeze()
        for l in model.model.layers[len(model.model.layers) - lora_layers :]:
            l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

        model.update(tree_unflatten(adapters))
        fused_linears = [(n, m.to_linear()) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
        model.update_modules(tree_unflatten(fused_linears))

        if self.config.de_quantize:
            de_quantize_layers = []
            for n, m in model.named_modules():
                if isinstance(m, nn.QuantizedLinear):
                    bias = "bias" in m
                    weight = m.weight
                    weight = mx.dequantize(weight, m.scales, m.biases, m.group_size, m.bits).astype(mx.float16)
                    output_dims, input_dims = weight.shape
                    linear = nn.Linear(input_dims, output_dims, bias=bias)
                    linear.weight = weight
                    if bias:
                        linear.bias = m.bias
                    de_quantize_layers.append((n, linear))
            model.update_modules(tree_unflatten(de_quantize_layers))

        weights = dict(tree_flatten(model.parameters()))
        if self.config.de_quantize:
            model_config.pop("quantization", None)
        save_model(save_path, weights, tokenizer, model_config)

        if self.config.upload_name is not None:
            hf_path = self.config.hf_path
            if not Path(model_path).exists():
                hf_path = model_path
            elif hf_path is None:
                raise ValueError("Must provide original Hugging Face repo to upload local model.")
            upload_to_hub(save_path, self.config.upload_name, hf_path)

