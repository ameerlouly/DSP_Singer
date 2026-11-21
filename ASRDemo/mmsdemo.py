from transformers import pipeline

# 1. Initialize the pipeline
# The project specifies facebook/mms-1b-all
model_id = "facebook/mms-1b-all"

print("Downloading and loading Meta MMS (this may take a few minutes)...")

# MMS requires you to specify the target language adapter.
# "ara" is the standard ISO code used for Arabic in this model family.
mms_pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    model_kwargs={"target_lang": "ara", "ignore_mismatched_sizes": True}
)

print("Model loaded. Transcribing...")


test_file = "G:\EECE\Year 4\DSP\DSP_Singer\ASRDemo\Tests\mixed_002.wav"
# 2. Transcribe the audio
# The pipeline automatically reads your .wav file and processes it
result = mms_pipe(test_file)

# 3. Print the output
print("\n--- Meta MMS Transcript ---")
print(result["text"])
