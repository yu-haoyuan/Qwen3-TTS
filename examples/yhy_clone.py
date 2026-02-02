import torch
import soundfile as sf
import os
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "/data/ptmodels/qwen3tts/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
)

ref_audio = "/data/xiaobu/all_tts/0123/Qwen3-TTS/assets/hxx_0113.wav"
ref_text  = "您现在的情况，我大概了解了，这些材料的话对我们后续的调解工作可能会有一定的帮助"

prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
)

# Read text lines
text_file = "/data/xiaobu/all_tts/0123/Qwen3-TTS/assets/text.txt"
with open(text_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
print(f"Total lines: {len(lines)}")

output_dir = "/data/xiaobu/all_tts/0123/Qwen3-TTS/assets"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")
for i, text in enumerate(lines):
    print(f"Generating {i+1}/{len(lines)}: {text}")
    wavs, sr = model.generate_voice_clone(
        text=[text],
        language=["Chinese"],
        voice_clone_prompt=prompt_items,
    )
    output_path = os.path.join(output_dir, f"{i}.wav")
    sf.write(output_path, wavs[0], sr)
    print(f"Saved to {output_path}")