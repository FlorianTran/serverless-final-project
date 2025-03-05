#!/usr/bin/env python3
import os
import argparse
import sys
import uuid
from gtts import gTTS
from pydub import AudioSegment
from inferrvc import RVC, load_torchaudio
import soundfile as sf

# Allow safe unpickling of the fairseq Dictionary class
import torch
from fairseq.data.dictionary import Dictionary

torch.serialization.add_safe_globals([Dictionary])


def text_to_speech(text, output_mp3):
    """Generate an MP3 file from text using gTTS."""
    tts = gTTS(text=text, lang="en")
    tts.save(output_mp3)
    print(f"TTS audio saved to: {output_mp3}")


def convert_mp3_to_wav(input_mp3, output_wav):
    """Convert an MP3 file to WAV format."""
    audio = AudioSegment.from_mp3(input_mp3)
    audio.export(output_wav, format="wav")
    print(f"Converted {input_mp3} --> {output_wav}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a .wav file from text using the Darth Vader RVC model"
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Text to convert to speech"
    )
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]  # Clean up any extra arguments

    # Set environment variables (forcing CPU inference and pointing to model directories)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["RVC_MODELDIR"] = "DarthVaderCustom"
    os.environ["RVC_INDEXDIR"] = "DarthVaderCustom"
    os.environ["RVC_OUTPUTFREQ"] = "45000"
    os.environ["RVC_RETURNBLOCKING"] = "True"

    # Define temporary file paths
    temp_mp3 = "temp.mp3"
    temp_wav = "temp.wav"

    # 1. Generate TTS audio from text
    text_to_speech(args.text, temp_mp3)

    # 2. Convert the MP3 to WAV
    convert_mp3_to_wav(temp_mp3, temp_wav)

    # 3. Load your RVC model (using files from the "DarthVaderCustom" folder)
    model_path = "DarthVaderCustom_e1000_s45000.pth"
    index_path = "added_IVF510_Flat_nprobe_1_DarthVaderCustom_v2.index"
    darth_vader = RVC(model_path, index=index_path)
    print("RVC model loaded:")
    print(darth_vader)
    print("Model name:", darth_vader.name)
    print("Model and index paths:", darth_vader.model_path, darth_vader.index_path)

    # 4. Load the generated WAV file using the provided utility
    aud, sr = load_torchaudio(temp_wav)
    print(f"Loaded audio with sample rate: {sr}")

    # 5. Run the inference/conversion.
    converted_audio = darth_vader(
        aud, 5, output_device="cpu", output_volume=RVC.MATCH_ORIGINAL, index_rate=0.9
    )

    # 6. Generate a random output filename and save the converted audio as a WAV file (44100 Hz)
    random_filename = f"{uuid.uuid4()}.mp3"
    sf.write(random_filename, converted_audio, 44100)
    print(f"Output WAV saved to: {random_filename}")

    # Cleanup temporary files
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    print("Temporary files cleaned up. Process finished successfully.")

    # Return (print) the random filename
    return random_filename


if __name__ == "__main__":
    filename = main()
