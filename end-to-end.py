# pip install realtimetts[gttsengine] pyttsx3 
# pip install RealtimeTTS
import requests
import json
from RealtimeTTS import TextToAudioStream, SystemEngine, GTTSEngine, GTTSVoice
import speech_recognition as sr

def listen(mic, recognizer):
    with mic:
        print("\nSay something!")
        audio_data = recognizer.listen(mic)
        print("Got it!")
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("\nCould not understand audio")
        return ""

    except sr.RequestError as e:
        print("\nCould not request results from Speech Recognition service; {0}".format(e))
        return ""

def printSameLine(str):
    print(str, end="", flush=True)

def createMessage(content, isUser=True):
    return {
        'role': 'user' if isUser else 'assistant',
        'content': content
    }

def sendMessage(endpoint, model, history, content, callback = None):
    msg = createMessage(content)
    history.append(msg)

    req = requests.post(endpoint, json={
        'model': model,
        'messages': history,
        'stream': True,
    }, stream=True)

    if req.status_code != 200:
        raise Exception(f"Failed to send message: {req.status_code} {req.reason}")

    req.raise_for_status()

    total_msg = ""
    for l in req.iter_lines():
        res_json = json.loads(l)

        if "error" in res_json: raise Exception(res_json.get('error'))

        res_msg = res_json.get('message').get('content')
        total_msg += res_msg

        if res_json.get('done'): 
            callback(res_msg, True)
            break

        if callback is not None: 
            callback(res_msg, False)

    history.append(createMessage(total_msg, False))
    return total_msg

def makeTTSCallback(stream, stop_char_list = [".", "!", "?", ","]):
    temp_sentence = ""
    def callback(token, is_done = False):
        nonlocal temp_sentence
        temp_sentence += token
        printSameLine(token)

        if is_done or any([c in token for c in stop_char_list]):
            stream.feed(temp_sentence)
            if not stream.is_playing():
                stream.play_async()
            temp_sentence = ""
            return
    return callback

def main():
    model = 'mistral-nemo'
    # LAN endpoint (run $env:OLLAMA_HOST="0.0.0.0" then ollama serve to expose)
    endpoint = 'http://10.1.1.238:11434/api/chat'
    history = []

    # use SystemEngine for offline TTS
    tts_stream = TextToAudioStream(SystemEngine())
    # for the translation pre prompt below
    # tts_stream = TextToAudioStream(GTTSEngine(GTTSVoice(language='es')))
    tts_callback = makeTTSCallback(tts_stream)

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 4000

    mic = sr.Microphone()
    with mic:
        recognizer.adjust_for_ambient_noise(mic)

    # TODO: implement realtime stt for translate sentence by sentence
    # without waiting for the user to finish speaking
    # not reliable for now
    # pre_prompt_condition = [
    #     'you are a translator',
    #     'i will speak in English',
    #     'you will speak in spanish',
    #     'you will only speak in spanish',
    #     'your responses will be the translation of what i say',
    # ]

    pre_prompt_condition = [
        'you are Nova',
        'a friendly and engaging conversation partner',
        'respond naturally and make the conversation flow smoothly',
        'feel free to ask questions, share your thoughts, and keep things lively and interactive, just like a real human would',
    ]

    print("\nNova:", end=" ")
    sendMessage(
        endpoint, 
        model, 
        history, 
        ". ".join(pre_prompt_condition),
        tts_callback)

    exit_word = "Nova exit"

    while True:
        # user_input = input("\nYou: ")
        # if user_input == "exit": 
        #     break

        print("\nSay '" + exit_word + "' to exit the conversation.")
        user_input = listen(mic, recognizer)
        print("\nYou:", user_input)

        if user_input == "":
            print("Failed to transcribe, please try again.")
            continue
        if exit_word.lower() in user_input.lower():
            break

        print("\nNova:", end=" ")
        sendMessage(
            endpoint, 
            model, 
            history, 
            user_input, 
            tts_callback)

    print("\nExiting...")

if __name__ == "__main__":
    main()