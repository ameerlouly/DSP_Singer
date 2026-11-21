from faster_whisper import WhisperModel

# 1. Load the model
# For a basic PC without a GPU, use device="cpu" and compute_type="int8"
# The project requires Whisper large-v3
model_size = "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

print("Model loaded. Transcribing...")

test_file = "Tests/mixed_002.wav"
# 2. Transcribe the audio
# The project processes Arabic videos, so setting language="ar" improves accuracy
segments, info = model.transcribe(test_file, language="ar")

# 3. Extract and print the text
# faster-whisper outputs a generator of segments, so you must loop through them
transcript_whisper = ""
for segment in segments:
    transcript_whisper += segment.text + " "


print("\n--- Faster-Whisper Transcript ---")
print(transcript_whisper.strip())
