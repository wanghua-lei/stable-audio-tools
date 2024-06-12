import json

with open("train.json", 'r') as f:
    data = json.load(f)

result=[]
for item in data:
    item['dataset'] = 'audiocaps'
    item['location'] = item['location']
    result.append(item)

with open('train.json', 'w') as f:
    json.dump(result, f, indent=4)


# while read line 
# do 
#     NEW_FILE="/mmu-audio-ssd/frontend/audioSep/wanghualei/code/speech/$(basename "$line" .wav)_44K.wav"
#     ffmpeg -i $line -ar 44100 -ac 1 "$NEW_FILE"  </dev/null

# done  <speech.txt
# head -n 100 original.txt > newfile.txt
