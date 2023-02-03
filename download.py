# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

import nemo.collections.asr as nemo_asr

def download_model():
    
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")

if __name__ == "__main__":
    download_model()