This application transcribes audio and video files to json or txt files.

I personally wanted to record Zoom calls and be able to have concise notes.

So I decided to create this application to take an input file whether being video or audio and turning it into it's transcript so that I could load it into gemini to prompt any questions I had about its content.

So what I did was using moviepy to handle the files, whether it's an audio or video file, I just extract the audio from it. Convert the audio to a numpy array then use whisper to transcribe it to a JSON or txt output.

Then I just the put the transcript file into gemini. I use gemini because it seems to have the biggest context window out of the models and I can generate accurate insights on the content of the input.