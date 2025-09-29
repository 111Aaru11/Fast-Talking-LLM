import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Adjust speaking rate and volume
engine.setProperty('rate', 130)   # speed (default ~200)
engine.setProperty('volume', 1.0) # max volume

# Save text to an audio file
text = "Imagine you are a wise traveler who has explored every corner of the world. Share with me a story about a lesson you learned from an unexpected encounter during one of your journeys."
engine.save_to_file(text, "moon_poem.wav")

# Run the speech engine to process and save
engine.runAndWait()

print("✅ File generated: moon_poem.wav")











# import pyttsx3

# # Initialize TTS engine
# engine = pyttsx3.init()

# # Adjust speaking rate and volume
# engine.setProperty('rate', 130)   # speed (default ~200)
# engine.setProperty('volume', 1.0) # max volume

# # Save text to an audio file
# text = ""
# engine.save_to_file(text, "moon_poem.wav")

# # Run the speech engine to process and save
# engine.runAndWait()

# print("✅ File generated: moon_poem.wav")
