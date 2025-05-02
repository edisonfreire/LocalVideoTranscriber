import os
import sys
import logging
import datetime
import json
import numpy as np
from moviepy import VideoFileClip, AudioFileClip, AudioClip
import whisper
from typing import Union

# --- Configuration ---
TARGET_SAMPLE_RATE = 16000 # Whisper expects 16kHz
OUTPUT_FORMAT = "txt"      # Choose desired output format: "json", "txt", "csv"
VOLUME_SCALE_FACTOR = 2.0   # Set to None or 1.0 to disable scaling, can be helpful with inputs with very quiet audio

# Define known extensions (can be expanded)
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

# --- Helper Functions ---

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS.fff format"""
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def ask_overwrite(filepath):
    """Asks the user if they want to overwrite an existing file."""
    if os.path.exists(filepath):
        response = input(f"Warning: Output file '{filepath}' already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logging.warning(f"User chose not to overwrite '{filepath}'. Aborting.")
            return False
    return True

def prepare_audio_array(audio_clip: AudioClip, target_sr=TARGET_SAMPLE_RATE) -> Union[np.ndarray, None]:
    """Converts a moviepy AudioClip to a mono, float32 NumPy array at target_sr."""
    try:
        logging.info(f"Converting audio clip to NumPy array ({target_sr}Hz, float32)...")
        # Use parameters to ensure resampling and correct byte depth for float32
        raw_audio_array = audio_clip.to_soundarray(fps=target_sr, nbytes=4, buffersize=2000)

        # Ensure the data type is float32 (important for Whisper)
        if raw_audio_array.dtype != np.float32:
            logging.warning(f"Audio array dtype is {raw_audio_array.dtype}, converting to float32.")
            raw_audio_array = raw_audio_array.astype(np.float32)

        # Ensure the audio is mono (Whisper expects 1D array)
        if raw_audio_array.ndim > 1 and raw_audio_array.shape[1] == 2: # Stereo
            logging.info("Audio is stereo, converting to mono by averaging channels.")
            mono_audio_array = raw_audio_array.mean(axis=1)
        elif raw_audio_array.ndim == 1: # Mono
            logging.info("Audio is mono.")
            mono_audio_array = raw_audio_array
        else: # Unexpected shape
            logging.error(f"Unexpected audio array shape: {raw_audio_array.shape}")
            return None

        logging.info(f"Audio array prepared: shape={mono_audio_array.shape}, dtype={mono_audio_array.dtype}")
        return mono_audio_array
    except Exception as e:
        logging.error(f"Error during audio array preparation: {e}", exc_info=True)
        return None

# --- Main script execution ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Get Input ---
    input_file_path = input("Enter video or audio file path (e.g., input/myvideo.mp4 or input/myaudio.wav): ")

    # --- Define Paths & Basic Validation ---
    input_dir = os.path.dirname(input_file_path) # Get dir from input
    input_filename = os.path.basename(input_file_path)
    output_dir = "output"
    base_input_name = os.path.splitext(input_filename)[0]
    output_transcript_path = os.path.join(output_dir, f"{base_input_name}_transcript.{OUTPUT_FORMAT}")

    logging.info(f"Checking for input file: {input_file_path}")
    if not os.path.isfile(input_file_path):
        logging.error(f"Error: Input file not found at '{input_file_path}'.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    if not ask_overwrite(output_transcript_path):
        sys.exit(0)

    # --- Detect File Type ---
    _, file_extension = os.path.splitext(input_filename)
    file_extension = file_extension.lower()
    file_type = None
    if file_extension in VIDEO_EXTENSIONS:
        file_type = 'video'
    elif file_extension in AUDIO_EXTENSIONS:
        file_type = 'audio'
    else:
        # Attempt to load with moviepy as a fallback? Or just fail.
        logging.warning(f"Warning: Unrecognized file extension '{file_extension}'. Attempting to process anyway...")
        # Let it try and fail later if moviepy can't handle it. Or exit:
        # logging.error(f"Error: Unsupported file extension '{file_extension}'.")
        # sys.exit(1)


    # --- Main Processing ---
    video_clip = None # Handle video resource
    audio_clip = None # Handle main audio resource (from video or audio file)
    mono_audio_array = None # To store the final array for Whisper

    try:
        # --- Load file based on type and get audio clip ---
        if file_type == 'video':
            logging.info(f"Processing as video file: {input_file_path}")
            video_clip = VideoFileClip(input_file_path)
            if video_clip.audio is None:
                raise ValueError("Video file has no audio track.")
            # Get the audio part from the video
            audio_clip = video_clip.audio

        elif file_type == 'audio':
            logging.info(f"Processing as audio file: {input_file_path}")
            # Load the audio file directly
            audio_clip = AudioFileClip(input_file_path)

        else: # Fallback for unrecognized extension if we didn't exit above
             logging.info(f"Attempting to load unrecognized file type with VideoFileClip: {input_file_path}")
             try:
                 # Try loading as video first
                 video_clip = VideoFileClip(input_file_path)
                 if video_clip.audio:
                     audio_clip = video_clip.audio
                     file_type = 'video' # Mark as video if successful
                     logging.info("Successfully loaded as video with audio track.")
                 else: # If it loaded but had no audio, try as pure audio
                    if video_clip: video_clip.close() # Close the videoclip handle first
                    video_clip = None
                    audio_clip = AudioFileClip(input_file_path)
                    file_type = 'audio' # Mark as audio
                    logging.info("Successfully loaded as audio file.")
             except Exception as load_err:
                 logging.error(f"Failed to load file '{input_file_path}' as either video or audio: {load_err}")
                 raise # Re-raise the error to be caught by the main handler


        # --- Process the obtained audio_clip ---
        if audio_clip is None:
             raise RuntimeError("Failed to extract or load audio clip.") # Should not happen if loading was successful

        # Apply optional volume scaling
        if VOLUME_SCALE_FACTOR is not None and VOLUME_SCALE_FACTOR != 1.0:
             logging.info(f"Applying volume scaling: x{VOLUME_SCALE_FACTOR}")
             audio_clip = audio_clip.with_volume_scaled(VOLUME_SCALE_FACTOR)

        # Prepare the NumPy array using the helper function
        mono_audio_array = prepare_audio_array(audio_clip, target_sr=TARGET_SAMPLE_RATE)

        if mono_audio_array is None:
            raise RuntimeError("Failed to prepare audio data array from the clip.")

        # --- Transcription ---
        logging.info("Loading Whisper model (base)...") # Consider making model name configurable
        model = whisper.load_model("base")

        logging.info(f"Starting transcription from audio data in memory...")
        result = model.transcribe(audio=mono_audio_array, verbose=True) # Pass NumPy array

        # --- Writing Transcript ---
        logging.info(f"Writing {OUTPUT_FORMAT.upper()} transcript to: {output_transcript_path}")
        with open(output_transcript_path, "w", encoding="utf-8") as f:
            # Choose output based on OUTPUT_FORMAT (using JSON example)
            if OUTPUT_FORMAT == "json":
                 output_data = {
                     "metadata": {
                         "source_file": input_filename,
                         "file_type_detected": file_type,
                         "model_used": "base",
                         "transcription_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         "detected_language": result.get("language")
                     },
                     "transcription_result": result
                 }
                 json.dump(output_data, f, ensure_ascii=False, indent=2)
            elif OUTPUT_FORMAT == "txt": # Timestamped text
                 f.write(f"# Transcription of: {input_filename}\n# Type: {file_type}\n# Model: base\n# Date: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n# Language: {result.get('language', 'unknown')}\n\n")
                 if "segments" in result:
                    for segment in result["segments"]:
                        f.write(f"[{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}] {segment['text'].strip()}\n")
                 else: f.write(result["text"])
            # Add elif for "csv", etc.
            else:
                 logging.warning(f"Unsupported output format '{OUTPUT_FORMAT}'. Saving raw text.")
                 f.write(result["text"])

        logging.info("Processing finished successfully.")

    except Exception as e:
        # General error catching
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
        sys.exit(1) # Exit with error status

    finally:
        # --- Resource Cleanup ---
        # Ensure clips opened in the try block are closed
        logging.info("Closing resources...")
        if audio_clip: # This holds the primary audio object
            try: audio_clip.close(); logging.info("Audio clip resource closed.")
            except Exception as e: logging.warning(f"Warning: Error closing audio clip resource: {e}")
        if video_clip: # This only exists if input was video
            try: video_clip.close(); logging.info("Video clip resource closed.")
            except Exception as e: logging.warning(f"Warning: Error closing video clip resource: {e}")