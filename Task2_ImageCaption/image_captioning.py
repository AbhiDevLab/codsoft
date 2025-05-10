import numpy as np
import os
import pickle
import gc
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from PIL import Image

# ========================
# CONFIGURATION
# ========================
MAX_CAPTION_LENGTH = 30
VOCAB_SIZE = 5000
EMBEDDING_DIM = 256
LSTM_UNITS = 512
BATCH_SIZE = 2
EPOCHS = 10

# ========================
# PATH SETUP (OS-INDEPENDENT)
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CAPTION_FILE = os.path.join(BASE_DIR, "captions.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "image_captioning_model.h5")
TOKENIZER_SAVE_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")

# ========================
# CORE FUNCTIONS (OPTIMIZED)
# ========================
def load_captions(caption_file):
    """Load captions with robust error handling"""
    captions = {}
    try:
        with open(caption_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',', 1)  # Split on first comma only
                if len(parts) == 2:
                    img_id = parts[0].strip()
                    caption = parts[1].strip()
                    captions.setdefault(img_id, []).append('startseq ' + caption.lower() + ' endseq')
    except Exception as e:
        print(f"Error loading captions: {str(e)}")
    return captions

def extract_features(image_dir):
    """Efficient feature extraction with VGG16"""
    vgg = VGG16(weights="imagenet", include_top=False, pooling='avg')
    feat_extractor = Model(inputs=vgg.input, outputs=vgg.output)
    features = {}
    
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_name)
            img_id = os.path.splitext(img_name)[0]
            try:
                img = img_to_array(load_img(img_path, target_size=(224, 224))) / 255.0
                features[img_id] = feat_extractor.predict(np.expand_dims(img, axis=0), verbose=0).flatten()
            except Exception as e:
                print(f"Skipping {img_name}: {str(e)}")
    
    gc.collect()  # Free memory
    return features

def data_generator(captions, features, tokenizer, batch_size=4):
    """Fixed generator that ensures proper batch generation"""
    # Create all possible training samples
    samples = []
    for img_id, caps in captions.items():
        if img_id in features:
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    samples.append((img_id, seq[:i], seq[i]))
    
    while True:
        # Shuffle samples at start of each epoch
        np.random.shuffle(samples)
        
        X1, X2, y = [], [], []
        for img_id, in_seq, out_word in samples:
            # Prepare inputs
            X1.append(features[img_id])
            X2.append(pad_sequences([in_seq], maxlen=MAX_CAPTION_LENGTH)[0])
            y.append(to_categorical([out_word], num_classes=vocab_size)[0])
            
            # Yield when batch is full
            if len(X1) == batch_size:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = [], [], []
        
        # Yield remaining samples if any
        if X1:
            yield ([np.array(X1), np.array(X2)], np.array(y))

def build_model():
    """Model with fixed dimensions and better initialization"""
    # Image branch (512-D from VGG pooling='avg')
    input1 = Input(shape=(512,))
    img_feat = Dense(512, activation='relu')(input1)
    
    # Text branch
    input2 = Input(shape=(MAX_CAPTION_LENGTH,))
    text_feat = Embedding(vocab_size, EMBEDDING_DIM)(input2)
    text_feat = LSTM(LSTM_UNITS, return_sequences=False,
                    kernel_initializer='he_normal')(text_feat)
    
    # Combined
    decoder = add([img_feat, text_feat])
    decoder = Dense(512, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder)
    
    model = Model(inputs=[input1, input2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def generate_caption(image_path, model, tokenizer):
    """Caption generation with error handling"""
    try:
        img = img_to_array(load_img(image_path, target_size=(224, 224))) / 255.0
        feat = VGG16(weights='imagenet', include_top=False, pooling='avg'
                    ).predict(np.expand_dims(img, axis=0)).flatten()
        
        caption = "startseq"
        for _ in range(MAX_CAPTION_LENGTH):
            seq = tokenizer.texts_to_sequences([caption])[0]
            seq = pad_sequences([seq], maxlen=MAX_CAPTION_LENGTH)
            pred = model.predict([np.array([feat]), seq], verbose=0)
            pred_word = tokenizer.index_word.get(np.argmax(pred), "")
            if pred_word == "endseq":
                break
            caption += " " + pred_word
        return caption.replace("startseq ", "").replace(" endseq", "")
    except Exception as e:
        return f"Caption Error: {str(e)}"

# ========================
# EXECUTION PIPELINE
# ========================
if __name__ == "__main__":
    # 1. Setup and Data Loading
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    print("1. Loading captions...")
    captions = load_captions(CAPTION_FILE)
    print(f"   Found {len(captions)} images with {sum(len(c) for c in captions.values())} captions")
    
    print("2. Extracting features...")
    features = extract_features(IMAGE_DIR)
    print(f"   Extracted features for {len(features)} images")
    
    # 2. Text Processing
    print("3. Preparing text data...")
    all_captions = [cap for caps in captions.values() for cap in caps]
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", filters='')
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"   Vocabulary size: {vocab_size}")
    
    # 3. Model Training
    print("4. Building model...")
    model = build_model()
    model.summary()
    
    print("5. Starting training...")

    # Verify generator output first
    try:
        test_batch = next(data_generator(captions, features, tokenizer, BATCH_SIZE))
        print("Generator test successful - batch shapes:")
        print(f"Images: {test_batch[0][0].shape}")
        print(f"Sequences: {test_batch[0][1].shape}")
        print(f"Targets: {test_batch[1].shape}")
    except Exception as e:
        print(f"Generator failed: {str(e)}")
        exit()

    # Calculate proper steps per epoch
    steps = max(1, sum(len(caps) for caps in captions.values()) // BATCH_SIZE)
    print(f"Training steps per epoch: {steps}")

    history = model.fit(
        data_generator(captions, features, tokenizer, BATCH_SIZE),
        steps_per_epoch=steps,
        epochs=min(EPOCHS, 3),  # Start with 3 epochs
        verbose=2
    )
    
    # 4. Save and Test
    print("6. Saving model...")
    model.save(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    test_image = os.path.join(IMAGE_DIR, "test_image.jpg")
    if os.path.exists(test_image):
        print(f"7. Testing on {os.path.basename(test_image)}...")
        print("Generated Caption:", generate_caption(test_image, model, tokenizer))
    
    print("\n=== COMPLETED ===")