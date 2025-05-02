from moviepy import VideoFileClip
import whisper
import logging
import os
import sys
import datetime

def format_timestamp(seconds):
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def ask_to_overwrite(filepath):
    if os.path.isfile(filepath):
        response = input(f"Do you want to overwrite {filepath} (y/n)").lower().strip()
        if response != "y":
            logging.warning(f"User chose not to overwrite {filepath}")
            return False
    return True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    video_name = input("Enter video name (e.g., myvideo.mov, myvideo.mp4): ")
    audio_name = input("Enter audio name (e.g., myaudio.mp3, myaudio.wav): ")
    logging.basicConfig(level=logging.INFO)

    input_dir = "input"
    output_dir = "output"
    input_video_path = os.path.join(input_dir, video_name)
    output_audio_path = os.path.join(output_dir, audio_name)
    base_video_name = os.path.splitext(video_name)[0]
    output_transcript_path = os.path.join(output_dir, f"{base_video_name}.txt")

    logging.info(f"Checking for input video: {input_video_path}")
    if not os.path.isfile(input_video_path):
        logging.error(f"Input video {input_video_path} does not exist")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    if not ask_to_overwrite(output_audio_path):
        sys.exit(0)
    if not ask_to_overwrite(output_transcript_path):
        sys.exit(0)

    video_clip = None
    audio_clip = None
    try:
        logging.info(f"Reading input video {input_video_path}")
        video_clip = VideoFileClip(input_video_path)

        logging.info(f"Extracting and scaling audio...")
        if video_clip.audio is None:
            logging.error(f"Input video {input_video_path} has no audio file")
            if video_clip:
                video_clip.close()
            sys.exit(1)

        audio_clip = video_clip.audio.with_volume_scaled(2.0)

        logging.info(f"Writing audio file to: {output_audio_path}")
        audio_clip.write_audiofile(output_audio_path)

        logging.info("Loading Whisper model (base)...")
        model = whisper.load_model("base")

        logging.info(f"Starting transcription for: {output_audio_path}")
        result = model.transcribe(audio=output_audio_path, verbose=True)

        logging.info(f"Writing transcript to: {output_transcript_path}")
        with open(output_transcript_path, "w", encoding="utf-8") as f:
            # Optional: Write some metadata header
            f.write(f"# Transcription of: {video_name}\n")
            f.write(f"# Model used: base\n")
            f.write(f"# Date: {datetime.datetime.now().isoformat()}\n")
            f.write(f"# Language: {result.get('language', 'unknown')}\n\n")

            # Write segments with timestamps
            if "segments" in result:
                for segment in result["segments"]:
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    # Optional: Include confidence if available/needed
                    # confidence = segment.get("confidence", "N/A")
                    # f.write(f"[{start_time} --> {end_time}] (Conf: {confidence:.2f}) {text}\n")
                    f.write(f"[{start_time} --> {end_time}] {text}\n")
            else:
                # Fallback if segments aren't available (shouldn't happen with default transcribe)
                f.write(result["text"])

        logging.info("Processing finished successfully.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.info("Closing resources...")
        if audio_clip:
            try:
                audio_clip.close()
                logging.info("Audio clip closed.")
            except Exception as e:
                logging.error(f"Error closing audio clip {audio_clip.name}")
        if video_clip:
            try:
                video_clip.close()
                logging.info("Video clip closed.")
            except Exception as e:
                logging.error(f"Error closing video clip {video_clip.name}")