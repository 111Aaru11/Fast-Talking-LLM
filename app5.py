import gradio as gr
from faster_whisper import WhisperModel
from ollama import chat
import pyttsx3
import logging

# -------------------------------
# Setup logging (file + console)
# -------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("loggings.log")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Application started.")

# -------------------------------
# Initialize models
# -------------------------------
try:
    # Whisper model (CTranslate2 backend, no PyTorch)
    whisper_model = WhisperModel("tiny", device="cuda", compute_type="float16")
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")

try:
    # TTS engine
    engine = pyttsx3.init()
    logger.info("TTS engine initialized.")
except Exception as e:
    logger.error(f"Error initializing TTS engine: {e}")

# -------------------------------
# Functions
# -------------------------------
def transcribe(audio_path):
    logger.info(f"Transcribing audio: {audio_path}")
    try:
        segments, _ = whisper_model.transcribe(audio_path, beam_size=1, vad_filter=True)
        transcript = " ".join([seg.text for seg in segments])
        logger.info(f"Transcription completed: {transcript}")
        return transcript
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return "[Transcription Error]"

def full_pipeline(audio):
    transcript = transcribe(audio)
    yield transcript, "🤔 Generating response...", None

    response_text = ""
    try:
        for chunk in chat(model="llama2:7b", messages=[{"role": "user", "content": transcript}], stream=True):
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                response_text += token
                yield transcript, response_text, None
        logger.info(f"LLM response generated: {response_text}")
    except Exception as e:
        logger.error(f"Error while streaming LLM response: {e}")

    # TTS output
    try:
        output_file = "response.wav"
        engine.save_to_file(response_text, output_file)
        engine.runAndWait()
        logger.info(f"TTS audio saved as {output_file}")
        yield transcript, response_text, output_file
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}")

# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## ⚡ Fast Talking LLM Assistant 🎤🤖 ")
    audio_input = gr.Audio(label="Speak/Upload", type="filepath")
    transcript_output = gr.Textbox(label="Transcribed Text")
    llm_output = gr.Textbox(label="LLM Response (streaming)")
    audio_output = gr.Audio(label="Response Audio", type="filepath")

    audio_input.change(
        fn=full_pipeline,
        inputs=audio_input,
        outputs=[transcript_output, llm_output, audio_output]
    )

logger.info("Launching Gradio app...")
demo.launch()












# import gradio as gr
# from faster_whisper import WhisperModel
# from ollama import chat
# import pyttsx3
# import logging
# import torch

# # -------------------------------
# # Setup logging (file + console)
# # -------------------------------
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# file_handler = logging.FileHandler("loggings.log")
# file_handler.setLevel(logging.INFO)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)

# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

# # if logger.hasHandlers():
# #     logger.handlers.clear()
# # logger.addHandler(file_handler)
# # logger.addHandler(console_handler)

# if not logger.hasHandlers():
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

# logger.info("Application started.")

# # -------------------------------
# # Check GPU availability
# # -------------------------------
# device = "cuda"
# logger.info(f"Using device: {device}")

# # -------------------------------
# # Initialize models
# # -------------------------------
# try:
#     # Whisper GPU model
#     whisper_model = WhisperModel("tiny", device=device, compute_type="float16")  # float16 recommended for GPU
#     logger.info("Whisper model loaded successfully on GPU.")
# except Exception as e:
#     logger.error(f"Error loading Whisper model: {e}")

# try:
#     # TTS engine
#     engine = pyttsx3.init()
#     logger.info("TTS engine initialized.")
# except Exception as e:
#     logger.error(f"Error initializing TTS engine: {e}")

# # -------------------------------
# # Functions
# # -------------------------------
# def transcribe(audio_path):
#     logger.info(f"Transcribing audio: {audio_path}")
#     try:
#         segments, _ = whisper_model.transcribe(audio_path, beam_size=1, vad_filter=True)
#         transcript = " ".join([seg.text for seg in segments])
#         logger.info(f"Transcription completed: {transcript}")
#         return transcript
#     except Exception as e:
#         logger.error(f"Error during transcription: {e}")
#         return "[Transcription Error]"

# def full_pipeline(audio):
#     transcript = transcribe(audio)
#     yield transcript, "🤔 Generating response...", None

#     response_text = ""
#     try:
#         for chunk in chat(model="llama2:7b", messages=[{"role": "user", "content": transcript}], stream=True):
#             if "message" in chunk and "content" in chunk["message"]:
#                 token = chunk["message"]["content"]
#                 response_text += token
#                 yield transcript, response_text, None
#         logger.info(f"LLM response generated: {response_text}")
#     except Exception as e:
#         logger.error(f"Error while streaming LLM response: {e}")

#     # TTS output
#     try:
#         output_file = "response.wav"
#         engine.save_to_file(response_text, output_file)
#         engine.runAndWait()
#         logger.info(f"TTS audio saved as {output_file}")
#         yield transcript, response_text, output_file
#     except Exception as e:
#         logger.error(f"Error in TTS generation: {e}")

# # -------------------------------
# # Gradio UI
# # -------------------------------
# with gr.Blocks() as demo:
#     gr.Markdown("## ⚡ Fast Talking LLM Assistant 🎤🤖 ")
#     audio_input = gr.Audio(label="Speak/Upload", type="filepath")
#     transcript_output = gr.Textbox(label="Transcribed Text")
#     llm_output = gr.Textbox(label="LLM Response (streaming)")
#     audio_output = gr.Audio(label="Response Audio", type="filepath")

# #     audio_input.upload(
# #     fn=full_pipeline,
# #     inputs=audio_input,
# #     outputs=[transcript_output, llm_output, audio_output]
# # )

#     audio_input.change(
#         fn=full_pipeline,
#         inputs=audio_input,
#         outputs=[transcript_output, llm_output, audio_output]
#     )

# logger.info("Launching Gradio app...")
# demo.launch()
