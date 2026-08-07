import os
import soundfile as sf
from typing import cast, Any
from kokoro import KPipeline
from pydub import AudioSegment
from pydub.generators import Sine

def generate_ding() -> AudioSegment:
    """Generates a simple 800 Hz sine wave 'ding' lasting 200ms."""
    duration_ms = 200  # Duration of the ding in milliseconds
    frequency = 1000   # Frequency of the ding in Hz
    ding = Sine(frequency).to_audio_segment(duration=duration_ms)
    return ding.fade_out(150)

def generate_podcast_audio(script_lines, output_filename="podcast_output.wav"):
    print("--- SYNTHESIZING AUDIO ---")

    pipeline = KPipeline(lang_code="a")

    voice_map = {
        "Host A": "af_sarah",
        "Host B": "am_michael"
    }

    final_audio = AudioSegment.empty()
    ding_sound = generate_ding()

    for i, line in enumerate(script_lines):
        print(f"Generating Kokoro audio for {line.speaker}...")

        # Get the assigned voice, default to 'af_heart' if speaker isn't mapped
        voice = voice_map.get(line.speaker, "af_heart")

        generator = pipeline(line.text, voice=voice, speed =1)

        for _, _, audio in generator:
            temp_filename = f"temp_line_{i}.wav"

            # Save the numpy array to a WAV file using soundfile at 24000Hz
            sf.write(temp_filename, cast(Any, audio), 24000)

            # Load it back into our pydub workflow
            line_audio = AudioSegment.from_file(temp_filename, format="wav")
            if line.category.value in ["VERBATIM_FACT", "INFERENCE"]: 
                final_audio += ding_sound
            final_audio += line_audio

            os.remove(temp_filename)  # Clean up the temporary file
        print(f"--- STITCHING COMPLETE ---")
    final_audio.export(output_filename, format="wav")
    print(f"Podcast saved locally to {output_filename}")

if __name__ == "__main__":
    # Mocking a script line to test the audio engine independently
    from evaluator import EvaluatedLine, LineCategory
    
    mock_lines = [
        EvaluatedLine(
            speaker="Host A", 
            text="Welcome to the Citation Soundboard.", 
            category=LineCategory.OPINION, 
            page_citation=None, 
            reasoning="Intro"
        ),
        EvaluatedLine(
            speaker="Host B", 
            text="Revenue increased by 45 percent.", 
            category=LineCategory.INFERENCE, 
            page_citation=1, 
            reasoning="Data point"
        )
    ]
    generate_podcast_audio(mock_lines, "test_podcast.wav")