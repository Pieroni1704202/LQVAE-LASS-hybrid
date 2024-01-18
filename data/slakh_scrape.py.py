import shutil
import yaml
import ffmpeg
from tqdm import tqdm
from pydub import AudioSegment, effects  
import numpy as np
import os 

def source_scarping(rootdir: str, sample_limit: int):
    '''Search in all the song files of slakh for the desired number of source files (bass/drums)'''

    # dict_midi_name = {}
    drums_counter = 0
    bass_counter = 0

    for entry in tqdm(os.scandir(rootdir)):

        # interrupt if reached desired sample num
        if bass_counter >=  sample_limit and drums_counter >= sample_limit:
            break

        if entry.is_dir():
            
            print(entry.name)

            with open(os.path.join( entry.path,'metadata.yaml'), 'r') as f:
                metadata = yaml.safe_load(f)
                for stem in metadata['stems']:
                    # check if there is a track of type drums inside the song
                    if metadata['stems'][stem]['inst_class'] == 'Drums' and metadata['stems'][stem]['audio_rendered']:
                        drums_counter += 1

                        src = os.path.join(entry.path, 'stems', stem + '.flac')
                        dst = os.path.join('.', 'sources', 'drums', entry.name + '_'+ str(drums_counter) + '.wav') 
                    
                        stream_drums = ffmpeg.input(src)
                        audio_drums = stream_drums.audio
                        stream_drums = ffmpeg.output(audio_drums, dst, **{'ar': '22050'})  # copy source at 22kHz
                        ffmpeg.run(stream_drums, capture_stdout=True, capture_stderr=True, overwrite_output=True)
                    
                    # check if there is a track of type bass inside the song
                    elif metadata['stems'][stem]['inst_class'] == 'Bass' and metadata['stems'][stem]['audio_rendered']:
                        bass_counter += 1

                        src = os.path.join(entry.path, 'stems', stem + '.flac')
                        dst = os.path.join('.', 'sources', 'bass', entry.name + '_'+ str(bass_counter) + '.wav') 

                        stream_bass = ffmpeg.input(src)
                        audio_bass = stream_bass.audio
                        stream_bass = ffmpeg.output(audio_bass, dst, **{'ar': '22050'}) # copy source at 22kHz
                        ffmpeg.run(stream_bass, capture_stdout=True, capture_stderr=True, overwrite_output=True)


    print(f"Total number of drums = {drums_counter}")
    print(f"Total number of bass = {bass_counter}")


def do_submix(mix_dir: str, source_1_path: str, source_2_path: str):
    '''Mix two source to form a mixture'''

    os.makedirs(mix_dir, exist_ok=True)

    scan_1 = os.scandir(source_1_path)
    scan_2 = os.scandir(source_2_path)

    
    for entry_1 in scan_1:
        if entry_1.is_file():
            
            entry_2 = next(scan_2)

            source_1 = AudioSegment.from_wav(entry_1.path)
            source_2 = AudioSegment.from_wav(entry_2.path)

            # overlay the two sources
            if len(source_1) > len(source_2):
                mixture = source_2.overlay(source_1)
            else:
                mixture= source_1.overlay(source_2)

            # normalize the mixture
            normalized_sound = effects.normalize(mixture)  

            frames_per_second = normalized_sound.frame_rate
            bytes_per_sample = normalized_sound.sample_width

            if bytes_per_sample != 2:
                print(f"ERROR: bytes_per_sample = {bytes_per_sample} for mixed track of {entry_1.name} and {entry_2.name}")
            if frames_per_second != 22050:
                print(f"ERROR: frames_per_second = {frames_per_second} for mixed track of {entry_1.name} and {entry_2.name}")
            
            first_name = entry_1.name.replace('.wav', '')

            normalized_sound.export(os.path.join(mix_dir, first_name + '_' + entry_2.name), format="wav")

   
def test(file_path: str):
    
    scan_2 = os.scandir(file_path)
    
    for entry_1 in scan_2:
        
        if 'flac' in entry_1.name:
            
            source_1 = AudioSegment.from_file(entry_1.path, format='flac', frame_rate=22050)

            print(source_1.frame_rate)
            print(source_1.sample_width)
            print(source_1.channels)

if __name__ == '__main__':

    # source_scarping('./slakh2100_flac_redux/train', 300)

    do_submix('./sources/mix', './sources/drums', './sources/bass')

    # test(file_path) #test the sample rate and channels of the audio