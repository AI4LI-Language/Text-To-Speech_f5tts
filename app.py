import spaces
import gradio as gr
from cached_path import cached_path
import tempfile

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
    save_spectrogram,
)


vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=str(
        cached_path("hf://toandev/F5-TTS-Vietnamese/model_latest.safetensors")
    ),
    vocab_file=str(cached_path("hf://toandev/F5-TTS-Vietnamese/vocab.txt")),
)


@spaces.GPU
def infer(ref_audio_orig: str, gen_text: str, speed: float = 1.0):
    if ref_audio_orig is None:
        raise gr.Error("Reference audio is required.")

    if gen_text is None or gen_text.strip() == "":
        raise gr.Error("Text to generate is required.")

    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            model,
            vocoder,
            cross_fade_duration=0.15,
            nfe_step=32,
            speed=speed,
        )

        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(combined_spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    except Exception as e:
        raise gr.Error(f"An error occurred during inference: {e}")


iface = gr.Interface(
    title="F5-TTS Vietnamese",
    description="Based on the [F5-TTS](https://github.com/SWivid/F5-TTS) model, a Diffusion Transformer with ConvNeXt V2, this Vietnamese text-to-speech model was trained on ~4 hours of Vietnamese audio data in 41k training steps. It boasts faster training and inference speeds, however, the quality of the synthesized speech may have noticeable imperfections such as choppiness or lack of natural intonation.",
    fn=infer,
    inputs=[
        gr.components.Audio(type="filepath", label="Reference Audio"),
        gr.components.Textbox(label="Text to Generate", lines=3),
        gr.components.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Adjust the speed of the audio.",
        ),
    ],
    outputs=[
        gr.components.Audio(type="numpy", label="Synthesized Audio"),
        gr.components.Image(type="filepath", label="Spectrogram"),
    ],
    submit_btn="Synthesize",
    clear_btn=None,
    flagging_mode="never",
    examples=[
        [
            "examples/01.wav",
            "Kiểm soát cảm xúc thực chất là một quá trình đánh giá lại bản thân, để tìm thấy tự do, thoát khỏi sự cuốn hút của chính bản ngã.",
            0.8,
        ],
        [
            "examples/02.wav",
            "Ngoài ra, nội dung ở bên kênh đấy tôi sẽ cố gắng là không nói bậy nhá.",
            1.0,
        ],
        [
            "examples/01.wav",
            "Cho tôi năm trăm triệu tôi sẽ gạch tên Pew và con tôi ra khỏi danh sách bạn bè, thực tế còn chịu tham gia một trận bốc xing để kết thúc tình nghĩa.",
            0.8,
        ],
    ],
)

if __name__ == "__main__":
    iface.queue().launch()
