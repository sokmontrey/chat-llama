# pip install realtimetts[gttsengine] pyttsx3 
# pip install RealtimeTTS
import requests
import json
from RealtimeTTS import TextToAudioStream, SystemEngine, GTTSEngine
import numpy as np

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
        if res_json.get('done'): break

        res_msg = res_json.get('message').get('content')
        total_msg += res_msg

        if callback is not None: callback(res_msg)

    history.append(createMessage(total_msg, False))
    return total_msg

def makeTTSCallback(stream, stop_char_list = [".", "!", "?", ","]):
    temp_sentence = ""
    def callback(token):
        nonlocal temp_sentence
        temp_sentence += token
        printSameLine(token)

        if any([c in token for c in stop_char_list]):
            stream.feed(temp_sentence)
            if not stream.is_playing():
                stream.play_async()
            temp_sentence = ""
            return

    return callback

def main():
    model = 'llama3'
    endpoint = 'http://localhost:11434/api/chat'
    history = []

    # use SystemEngine for system TTS most likely pyttsx3
    # can also use OpenAI voice service with their API
    tts_stream = TextToAudioStream(GTTSEngine())
    tts_callback = makeTTSCallback(tts_stream)

    pre_prompt_condition = [
        # 'your name is Nova',
        'completely verbal conversation',
        'short',
        'professional but casual',
    ]

    print("""type:
"exit"  : exit
"pause" : interupt the speaking
"stop"  : stop the speaking""")

    print("\nNova:", end=" ")
    sendMessage(
        endpoint, 
        model, 
        history, 
        "Condition for this conversation: " + ", ".join(pre_prompt_condition),
        tts_callback)
    print("\nReady...")

    is_speaking = True
    while True:
        user_input = input("\nYou: ")
        if user_input == "":
            print("Failed to transcribe, please try again.")
            continue
        if user_input == "exit": 
            break
        if user_input == "pause": 
            tts_stream.stop()
            continue
        if user_input == "stop": 
            is_speaking = False
            continue

        print("\nNova:", end=" ")
        sendMessage(
            endpoint, 
            model, 
            history, 
            user_input, 
            tts_callback if is_speaking else printSameLine)

    print("\nExiting...")

if __name__ == "__main__":
    main()