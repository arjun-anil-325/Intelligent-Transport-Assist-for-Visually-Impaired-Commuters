def give_me_speech(text):
    # Load model and tokenizer
    model = VitsModel.from_pretrained("aoxo/swaram")
    tokenizer = AutoTokenizer.from_pretrained("aoxo/swaram")

    # Create the full text
    a = "അടുത്ത ബസ് "
    b = " വരെ പോകും."
    fulltext = a + text + b
    print(fulltext)

    # Tokenize the input text
    inputs = tokenizer(fulltext, return_tensors="pt")

    # Generate audio
    with torch.no_grad():
        output = model(**inputs).waveform

    # Process the output
    print("Output shape:", output.shape)
    output = output.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to NumPy
    output = output.astype('float32')         # Ensure data type is float32

    # Save the audio for playback
    sf.write("output.wav", output, samplerate=model.config.sampling_rate)
    print("Audio saved as output.wav")

    # Return the path to the saved audio file
    return "output.wav"

from transformers import AutoTokenizer
from transformers import VitsModel # Replace 'some_library' with the actual library providing VitsModel
import torch
import soundfile as sf




give_me_speech("അടുത്ത")