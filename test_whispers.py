from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu")  # or "cuda" for GPU
segments, info = model.transcribe("output.wav")
for segment in segments:
    print(segment.text)
