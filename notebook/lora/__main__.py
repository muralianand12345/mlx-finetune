import argparse

from core import LORA, LORAConfig, FUSE, FUSEConfig

def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA Fine-Tuning")
    parser.add_argument("--model", default="mlx_model", help="The path to the local model directory or HuggingFace model repo")
    parser.add_argument("--max-tokens", "-m", type=int, default=100, help="The maximum number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="The prompt to use for generation")

    parser.add_argument("--train", action="store_true", help="Do training")
    parser.add_argument("--add-eos-token", type=int, default=1, help="Enable add_eos_token for tokenizer")
    parser.add_argument("--data", type=str, default="data/", help="Directory containing the {train, valid, test}.jsonl files")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of layers to apply LoRA to")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--val-batches", type=int, default=25, help="Number of validation batches, -1 uses the entire validation set")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Adam learning rate")
    parser.add_argument("--steps-per-report", type=int, default=10, help="Number of training steps between loss reporting")
    parser.add_argument("--steps-per-eval", type=int, default=200, help="Number of training steps between validations")
    parser.add_argument("--resume-adapter-file", type=str, default=None, help="Path to a LoRA adapter file to resume training from")
    parser.add_argument("--adapter-file", type=str, default="adapters.npz", help="Path to save/load the trained LoRA adapter file")
    parser.add_argument("--save-every", type=int, default=100, help="Save the model every N iterations")
    parser.add_argument("--test", action="store_true", help="Run test set evaluation after training")
    parser.add_argument("--test-batches", type=int, default=500, help="Number of test batches, -1 uses the entire test set")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    parser.add_argument("--fuse", action="store_true", help="Fuse the LoRA adapters into the base model and save to HuggingFace format")
    parser.add_argument("--save-path", type=str, default="fused_model", help="Path to save the fused model")
    parser.add_argument("--hf-path", type=str, default=None, help="HuggingFace model repo path to upload the fused model to")
    parser.add_argument("--upload-name", type=str, default=None, help="If specified, upload the fused model to HuggingFace with this repo name")
    parser.add_argument("--de-quantize", action="store_true", help="De-quantize the model when fusing LoRA adapters (for QLoRA models)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    lora_config = LORAConfig(
        train=args.train,
        lora_layers=args.lora_layers,
        prompt=args.prompt,
        temp=args.temp,
        max_tokens=args.max_tokens,
        learning_rate=args.learning_rate,
        iterations=args.iters,
        val_batches=args.val_batches,
        batch_size=args.batch_size,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        resume_adapter_file=args.resume_adapter_file,
        adapter_file=args.adapter_file,
        save_every=args.save_every,
        test=args.test,
        test_batches=args.test_batches,
        seed=args.seed,
    )

    lora_trainer = LORA(config=lora_config)
    lora_trainer.invoke(model_path=args.model, data=args.data)

    if args.fuse:
        fuse_config = FUSEConfig(
            adapter_file=args.adapter_file,
            hf_path=args.hf_path,
            upload_name=args.upload_name,
            de_quantize=args.de_quantize,
        )

        fuse_runner = FUSE(config=fuse_config)
        fuse_runner.invoke(model_path=args.model, save_path=args.save_path)

if __name__ == "__main__":
    main()

    #uv run python lora --model mistralai/Mistral-7B-Instruct-v0.2 --data ./model_a_1/data --train --batch-size 1 --lora-layers 4 --adapter-file adapters_mistral.npz --fuse --save-path ./model_a_1/fused_mistral_model 