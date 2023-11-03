import sounddevice
from whisper_mic.whisper_mic import WhisperMic

pause = 2  # Pause time before entry ends

mic = WhisperMic()

while True:
    result = mic.listen()
    # mic.listen_loop()
    print(result)
