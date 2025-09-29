from TTS.api import TTS

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")  # load an English voice model
tts.tts_to_file("Hello, how are you?", file_path="outputs.wav")
