import gradio as gr
import numpy as np
import librosa
from test_whispers import WhisperModel
from TTS.api import TTS
from ollama import chat

# -------------------------------
# Initialize models
# -------------------------------

# Whisper model for speech-to-text
whisper_model = WhisperModel("medium", device = "cpu")  # or "medium" for smaller model

# Coqui TTS model for text-to-speech
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)

# -------------------------------
# Functions
# -------------------------------

def transcribe(audio):
    """Convert audio to text using Faster Whisper"""
    if audio is None or len(audio) == 0:
        return "No audio detected"

    # audio is just a numpy array of samples from Gradio
    samples = np.array(audio, dtype=np.float32)

    # If stereo, convert to mono
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    # Resample to 16kHz
    samples = librosa.resample(samples, orig_sr=16000, target_sr=16000)

    segments, _ = whisper_model.transcribe(samples, sample_rate=16000)
    text = " ".join([seg.text for seg in segments])
    return text


# def transcribe(audio):
#     """Convert audio to text using Faster Whisper"""
#     if audio is None or len(audio) == 0:
#         return "No audio detected"

#     # Gradio returns (samples, sr)
#     if isinstance(audio, tuple) or isinstance(audio, list):
#         samples, sr = audio
#         samples = np.array(samples, dtype=np.float32)
#         # If stereo, convert to mono
#         if samples.ndim > 1:
#             samples = samples.mean(axis=1)
#         # Resample to 16kHz
#         samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
#     else:
#         samples = np.array(audio, dtype=np.float32)
#         if samples.ndim > 1:
#             samples = samples.mean(axis=1)
    
#     segments, _ = whisper_model.transcribe(samples, sample_rate=16000)
#     text = " ".join([seg.text for seg in segments])
#     return text


def generate_response(text_input):
    """Get LLM response from Ollama"""
    if not text_input:
        return ""
    response = chat(model="llama2", messages=[{"role": "user", "content": text_input}])
    # Ollama response is usually a dict with 'content'

    return response['message']['content'] #new line
    # if isinstance(response, dict):
    #     return response.get("content", "")
    # return str(response)

def speak_text(text):
    """Convert text to speech using Coqui TTS"""
    if not text:
        return None
    output_file = "output.wav"
    try:
        # tts = TTS("tts_models/en/ljspeech/tacotron2-DDC") #new line 
        tts.tts_to_file(text=text, file_path=output_file)
    except Exception as e:
        print("TTS Error:", e)
        return None
    return output_file

def full_pipeline(audio):
    """Full pipeline: speech -> LLM -> speech"""
    try:
        transcript = transcribe(audio)
    except Exception as e:
        transcript = f"Transcription error: {e}"
    try:
        response = generate_response(transcript)
    except Exception as e:
        response = f"LLM error: {e}"
    try:
        audio_file = speak_text(response)
    except Exception as e:
        audio_file = None
        print("TTS error:", e)
    return transcript, response, audio_file

# -------------------------------
# Gradio Interface
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Talking LLM Assistant 🎤🤖")
    
    audio_input = gr.Audio(label="Speak something", type="numpy")
    transcript_output = gr.Textbox(label="Transcribed Text")
    llm_response_output = gr.Textbox(label="LLM Response")
    audio_output = gr.Audio(label="Response Audio")
    
    audio_input.change(fn=full_pipeline,
                       inputs=audio_input,
                       outputs=[transcript_output, llm_response_output, audio_output])

demo.launch()



















# import gradio as gr
# import numpy as np
# import librosa
# from faster_whisper import WhisperModel
# from TTS.api import TTS
# from ollama import chat

# # -------------------------------
# # Initialize models
# # -------------------------------

# # Whisper model for speech-to-text
# whisper_model = WhisperModel("large-v2")  # can use smaller models if needed

# # Coqui TTS model for text-to-speech
# tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)

# # -------------------------------
# # Functions
# # -------------------------------

# def transcribe(audio):
#     """Convert audio to text using Faster Whisper"""
#     if audio is None:
#         return ""
    
#     # Gradio returns (samples, sr) for type="numpy"
#     if isinstance(audio, tuple) or isinstance(audio, list):
#         audio_samples, sr = audio
#         # Convert to float32
#         audio_samples = np.array(audio_samples, dtype=np.float32)
#         # Resample to 16kHz (Whisper expects this)
#         audio_samples = librosa.resample(audio_samples, orig_sr=sr, target_sr=16000)
#     else:
#         audio_samples = np.array(audio, dtype=np.float32)
    
#     segments, _ = whisper_model.transcribe(audio_samples, sample_rate=16000)
#     text = " ".join([segment.text for segment in segments])
#     return text

# def generate_response(text_input):
#     """Get LLM response from Ollama"""
#     if not text_input:
#         return ""
#     response = chat(model="llama2", messages=[{"role": "user", "content": text_input}])
#     return response.get("content", "")  # sometimes response is a dict

# def speak_text(text):
#     """Convert text to speech using Coqui TTS"""
#     if not text:
#         return None
#     output_file = "output.wav"
#     tts.tts_to_file(text=text, file_path=output_file)
#     return output_file

# def full_pipeline(audio):
#     """Full pipeline: speech -> LLM -> speech"""
#     transcript = transcribe(audio)
#     response = generate_response(transcript)
#     audio_file = speak_text(response)
#     return transcript, response, audio_file

# # -------------------------------
# # Gradio Interface
# # -------------------------------
# with gr.Blocks() as demo:
#     gr.Markdown("## Talking LLM Assistant 🎤🤖")
    
#     # Audio input from microphone
#     audio_input = gr.Audio(label="Speak something", type="numpy")
    
#     # Text outputs
#     transcript_output = gr.Textbox(label="Transcribed Text")
#     llm_response_output = gr.Textbox(label="LLM Response")
    
#     # Audio output
#     audio_output = gr.Audio(label="Response Audio")
    
#     # Connect inputs and outputs
#     audio_input.change(fn=full_pipeline,
#                        inputs=audio_input,
#                        outputs=[transcript_output, llm_response_output, audio_output])

# # Launch interface
# demo.launch()











# # import gradio as gr
# # import numpy as np
# # from faster_whisper import WhisperModel
# # from TTS.api import TTS
# # from ollama import chat

# # # -------------------------------
# # # Initialize models
# # # -------------------------------

# # # Whisper model for speech-to-text
# # whisper_model = WhisperModel("large-v2")  # Use smaller models like "medium" if needed

# # # Coqui TTS model for text-to-speech
# # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)

# # # -------------------------------
# # # Functions
# # # -------------------------------

# # def transcribe(audio):
# #     """Convert audio to text using Faster Whisper"""
# #     if audio is None:
# #         return ""
# #     audio = np.array(audio, dtype=np.float32)
# #     segments, _ = whisper_model.transcribe(audio)
# #     text = " ".join([segment.text for segment in segments])
# #     return text

# # def generate_response(text_input):
# #     """Get LLM response from Ollama"""
# #     if not text_input:
# #         return ""
# #     response = chat(model="llama2", messages=[{"role": "user", "content": text_input}])
# #     return response

# # def speak_text(text):
# #     """Convert text to speech using Coqui TTS"""
# #     if not text:
# #         return None
# #     output_file = "output.wav"
# #     tts.tts_to_file(text=text, file_path=output_file)
# #     return output_file

# # def full_pipeline(audio):
# #     """Full pipeline: speech -> LLM -> speech"""
# #     transcript = transcribe(audio)
# #     response = generate_response(transcript)
# #     audio_file = speak_text(response)
# #     return transcript, response, audio_file

# # # -------------------------------
# # # Gradio Interface
# # # -------------------------------
# # with gr.Blocks() as demo:
# #     gr.Markdown("## Talking LLM Assistant 🎤🤖")
    
# #     # Audio input from microphone
# #     audio_input = gr.Audio(label="Speak something", type="numpy")
    
# #     # Text outputs
# #     transcript_output = gr.Textbox(label="Transcribed Text")
# #     llm_response_output = gr.Textbox(label="LLM Response")
    
# #     # Audio output
# #     audio_output = gr.Audio(label="Response Audio")
    
# #     # Connect inputs and outputs
# #     audio_input.change(fn=full_pipeline,
# #                        inputs=audio_input,
# #                        outputs=[transcript_output, llm_response_output, audio_output])

# # # Launch interface
# # demo.launch()


# # # import gradio as gr

# # # import sounddevice as sd
# # # import numpy as np
# # # from faster_whisper import WhisperModel
# # # from TTS.api import TTS
# # # from ollama import chat

# # # # -------------------------------
# # # # Initialize models
# # # # -------------------------------

# # # # Whisper model for speech-to-text
# # # whisper_model = WhisperModel("large-v2")  # you can use smaller models like "medium" if needed

# # # # Coqui TTS model for text-to-speech
# # # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)

# # # # Ollama LLM
# # # # ollama_client = OllamaClient()

# # # # -------------------------------
# # # # Functions
# # # # -------------------------------

# # # def transcribe(audio):
# # #     """Convert audio to text using Faster Whisper"""
# # #     if audio is None:
# # #         return ""
# # #     audio = np.array(audio, dtype=np.float32)
# # #     segments, _ = whisper_model.transcribe(audio)
# # #     text = " ".join([segment.text for segment in segments])
# # #     return text

# # # def generate_response(text_input):
# # #     """Get LLM response from Ollama"""
# # #     if not text_input:
# # #         return ""
# # #     response = chat("llama2", text_input)
# # #     return response

# # # def speak_text(text):
# # #     """Convert text to speech using Coqui TTS"""
# # #     if not text:
# # #         return None
# # #     output_file = "output.wav"
# # #     tts.tts_to_file(text=text, file_path=output_file)
# # #     return output_file

# # # def full_pipeline(audio):
# # #     """Full pipeline: speech -> LLM -> speech"""
# # #     transcript = transcribe(audio)
# # #     response = generate_response(transcript)
# # #     audio_file = speak_text(response)
# # #     return transcript, response, audio_file

# # # # -------------------------------
# # # # Gradio Interface
# # # # -------------------------------
# # # with gr.Blocks() as demo:
# # #     gr.Markdown("## Talking LLM Assistant 🎤🤖")
    
# # #     with gr.Row():
# # #         audio_input = gr.Audio(label="Speak something", type="numpy")
    
# # #     with gr.Row():
# # #         transcript_output = gr.Textbox(label="Transcribed Text")
# # #         llm_response_output = gr.Textbox(label="LLM Response")
    
# # #     audio_output = gr.Audio(label="Response Audio")
    
# # #     audio_input.change(fn=full_pipeline,
# # #                        inputs=audio_input,
# # #                        outputs=[transcript_output, llm_response_output, audio_output])

# # # # Launch the interface
# # # demo.launch()
