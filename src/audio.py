import os
import pyttsx3
from typing import cast
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

    final_audio = AudioSegment.empty()
    ding_sound = generate_ding()

    for i, line in enumerate(script_lines):
        print(f"Generating audio for {line.speaker}...")

        engine = pyttsx3.init()
        voices = cast(list, engine.getProperty('voices'))

        voice_map = {
                "Host A": voices[0].id if len(voices) > 0 else None,
                "Host B": voices[1].id if len(voices) > 1 else voices[0].id
        }

        engine.setProperty('voice', voice_map.get(line.speaker, voices[0].id))

        temp_filename = f"temp_line_{i}.wav"
        engine.save_to_file(line.text, temp_filename)
        engine.runAndWait()

        del engine  # Free up resources

        line_audio = AudioSegment.from_file(temp_filename, format="wav")
        if line.category.value in ["VERBATIM_FACT", "INFERENCE"]: 
            final_audio += ding_sound

        final_audio += line_audio

        os.remove(temp_filename)

    print(f"--- STITCHING COMPLETE ---")
    # Export the final combined audio
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