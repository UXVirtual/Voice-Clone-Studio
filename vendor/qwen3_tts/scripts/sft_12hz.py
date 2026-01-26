# coding=utf-8
import argparse
import json
import os
import shutil
import torch
import sys

# Add vendor root to path to find dataset module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from accelerate import Accelerator
from qwen3_tts.dataset.dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from huggingface_hub import snapshot_download

target_speaker_embedding = None

def train():
    global target_speaker_embedding
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="custom_speaker")
    args = parser.parse_args()

    # Log to tensorboard in output dir
    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard", project_dir=args.output_model_path)
    
    MODEL_PATH = args.init_model_path
    
    # Load Model
    # Since we need to modify the model structure for fine-tuning embeddings if needed, 
    # but the script assumes we just load it.
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    with open(args.train_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    train_data = [json.loads(line.strip()) for line in lines if line.strip()]
    
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        print(f"TRAIN_INFO: steps_per_epoch={len(train_dataloader)}")
        print(f"TRAIN_INFO: total_epochs={args.num_epochs}")

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                
                # Assign speaker embedding to special token pos (idx 6) - logic from Qwen script
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states[0][-1]
                
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]
                
                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                
                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            # Save output to a temporary local directory first to avoid IO errors on mounted volumes (Windows/WSL2)
            # Create a localized temp folder that is NOT on the mounted volume
            # os.path.dirname(args.output_model_path) is likely on volume, so use /tmp or purely python tempfile
            import tempfile
            
            # Create a temporary directory structure
            # We want to name the final folder "checkpoint-epoch-{epoch}"
            ckpt_name = f"checkpoint-epoch-{epoch}"
            
            # Use a context manager for temporary directory
            with tempfile.TemporaryDirectory() as temp_base_dir:
                 temp_output_dir = os.path.join(temp_base_dir, ckpt_name)
                 os.makedirs(temp_output_dir, exist_ok=True)
                 
                 final_output_dir = os.path.join(args.output_model_path, ckpt_name)
                 if os.path.exists(final_output_dir):
                     shutil.rmtree(final_output_dir)
                 os.makedirs(final_output_dir, exist_ok=True)

                 # 1. Prepare Content in Temp Dir
                 if os.path.isdir(MODEL_PATH):
                     # Copy only configuration and tokenizer files to save space using ignore patterns
                     shutil.copytree(MODEL_PATH, temp_output_dir, dirs_exist_ok=True, 
                                     ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pt", "*.pth", "*.msgpack", "*.h5"))
                 else:
                     # It's an HF repo ID, download configs from Hub
                     try:
                         snapshot_download(
                             repo_id=MODEL_PATH, 
                             local_dir=temp_output_dir, 
                             allow_patterns=["*.json", "speech_tokenizer/*", "text_tokenizer/*", "tokenizer_config.json", "vocab.json", "merges.txt"],
                             local_dir_use_symlinks=False
                         )
                     except Exception as e:
                         accelerator.print(f"Warning: Failed to download config files from Hub: {e}")

                 # Update Config
                 config_dict = config.to_dict()
                 
                 # Sanitize: Remove model_type from nested configs to avoid init errors
                 if "speaker_encoder_config" in config_dict and isinstance(config_dict["speaker_encoder_config"], dict):
                     config_dict["speaker_encoder_config"].pop("model_type", None)
                 
                 config_dict["tts_model_type"] = "custom_voice"
                 talker_config = config_dict.get("talker_config", {})
                 if isinstance(talker_config, dict):
                      talker_config.pop("model_type", None)
                      
                 # Use lowercase speaker name for config keys to ensure matching during inference
                 # (Qwen3TTSForConditionalGeneration does `speaker.lower() in spk_id`)
                 speaker_key = args.speaker_name.lower()
                 talker_config["spk_id"] = {speaker_key: 3000}
                 talker_config["spk_is_dialect"] = {speaker_key: False}
                 config_dict["talker_config"] = talker_config
                 
                 with open(os.path.join(temp_output_dir, "config.json"), 'w', encoding='utf-8') as f:
                     json.dump(config_dict, f, indent=2, ensure_ascii=False)

                 # Save Weights
                 unwrapped_model = accelerator.unwrap_model(model)
                 state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

                 drop_prefix = "speaker_encoder"
                 keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
                 for k in keys_to_drop:
                     del state_dict[k]

                 weight = state_dict['talker.model.codec_embedding.weight']
                 # Ensure index 3000 is valid or this will raise IndexError if vocab is smaller
                 # We assume Qwen3-TTS base model has large enough vocab/embedding
                 if 3000 < weight.shape[0]:
                     state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
                 
                 # Save SafeTensors to TEMP dir first (local container FS, reliable IO)
                 save_file(state_dict, os.path.join(temp_output_dir, "model.safetensors"))
                 
                 # 2. Move to Final Volume Location
                 # We copy individual files to avoid permissions/locking issues with moving folders on some mounts
                 for item in os.listdir(temp_output_dir):
                     s = os.path.join(temp_output_dir, item)
                     d = os.path.join(final_output_dir, item)
                     if os.path.isdir(s):
                         shutil.copytree(s, d, dirs_exist_ok=True)
                     else:
                         shutil.copy2(s, d)
                         
                 accelerator.print(f"Saved checkpoint to {final_output_dir}")

if __name__ == "__main__":
    train()
