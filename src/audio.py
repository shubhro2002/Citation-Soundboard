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

    # Provide a pool of distinct Kokoro voices
    available_voices = ["af_sarah", "am_michael", "af_bella", "am_adam", "af_nicole"]
    
    # Start with an empty map
    dynamic_voice_map = {}

    final_audio = AudioSegment.empty()
    ding_sound = generate_ding()

    for i, line in enumerate(script_lines):
        speaker = line.get('speaker', 'Host A')
        text = line.get('text', '')
        category = line.get('step_2_category', 'OPINION')

        # If we haven't seen this speaker before, assign them a unique voice
        if speaker not in dynamic_voice_map:
            if len(available_voices) > 0:
                # Pop the first voice from the available list
                dynamic_voice_map[speaker] = available_voices.pop(0)
            else:
                # Safe fallback if there are more than 5 speakers
                dynamic_voice_map[speaker] = "af_heart" 
                
        voice = dynamic_voice_map[speaker]

        print(f"Generating Kokoro audio for {speaker} (Voice: {voice})...")

        generator = pipeline(text, voice=voice, speed=1)

        for _, _, audio in generator:
            temp_filename = f"temp_line_{i}.wav"

            # Save the numpy array to a WAV file using soundfile at 24000Hz
            sf.write(temp_filename, cast(Any, audio), 24000)

            # Load it back into our pydub workflow
            line_audio = AudioSegment.from_file(temp_filename, format="wav")
            
            # 2. UPDATED: Check the newly mapped 'category' variable for the ding
            if category in ["VERBATIM_FACT", "INFERENCE"]: 
                final_audio += ding_sound
            final_audio += line_audio

            os.remove(temp_filename)  # Clean up the temporary file
            
        print(f"--- STITCHING COMPLETE ---")
        
    final_audio.export(output_filename, format="wav")
    print(f"Podcast saved locally to {output_filename}")

if __name__ == "__main__":
    # Mocking a script line to test the audio engine independently
    from evaluator import EvaluatedLine
    
    mock_lines = [
        EvaluatedLine(
            speaker="Host A", 
            text="Welcome to the Citation Soundboard.", 
            step_2_category="OPINION", 
            step_3_page_citation=None, 
            step_1_explanation_sentence="Intro"
        ).dict(),
        EvaluatedLine(
            speaker="Host B", 
            text="Revenue increased by 45 percent.", 
            step_2_category="INFERENCE", 
            step_3_page_citation=1, 
            step_1_explanation_sentence="Data point"
        ).dict()
    ]
    generate_podcast_audio(mock_lines, "test_podcast.wav")