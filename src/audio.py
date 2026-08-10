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

    master_speech = AudioSegment.empty()
    ding_sound = generate_ding()
    short_pause = AudioSegment.silent(duration=600)

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
            sf.write(temp_filename, cast(Any, audio), 24000)
            line_audio = AudioSegment.from_file(temp_filename, format="wav")
            
            if category in ["VERBATIM_FACT", "INFERENCE"]: 
                master_speech += ding_sound
                
            master_speech += line_audio
            os.remove(temp_filename)  # Clean up the temporary file

        master_speech += short_pause    
        print(f"--- STITCHING COMPLETE ---")
        
    print(f"\n--- APPLYING POST-PRODUCTION ---")

    intro_padding = AudioSegment.silent(duration=2000)
    outro_padding = AudioSegment.silent(duration=3000)
    master_speech = intro_padding + master_speech + outro_padding

    bg_music_path = "src/bg_music.mp3"
    if os.path.exists(bg_music_path):
        print("  -> Found 'bg_music.mp3'. Mixing and applying dynamic ducking...")
        bg_music = AudioSegment.from_file(bg_music_path)

        # Loop music if it's shorter than the podcast length
        loops_needed = len(master_speech) // len(bg_music) + 1
        bg_music = bg_music * loops_needed
        bg_music = bg_music[:len(master_speech)] # Trim to exact podcast length

        # Slice the music into 3 parts and adjust dB levels
        intro_music = bg_music[:2000] - 5      # 2 seconds intro at normal volume
        speech_music = bg_music[2000:-3000] - 18 # Drop volume heavily while hosts speak
        outro_music = bg_music[-3000:] - 5     # 3 seconds outro fade back up

        ducked_bg = intro_music + speech_music + outro_music
        ducked_bg = ducked_bg.fade_in(1000).fade_out(2000)

        final_audio = master_speech.overlay(ducked_bg)
    else:
        print("  -> No 'bg_music.mp3' found. Skipping background track.")
        final_audio = master_speech

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