import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import tensorflow as tf

class FaceRecognizer:
    """Class for face recognition using various methods with GPU acceleration."""
    
    def __init__(self, method='eigenfaces', use_gpu=True):
        """
        Initialize the face recognizer.
        
        Args:
            method (str): Recognition method ('eigenfaces', 'lbph', 'ml', or 'deep')
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.method = method
        self.use_gpu = use_gpu
        self.face_size = (100, 100)  # Standard size for face images
        
        # Check if TensorFlow can access the GPU
        if self.use_gpu:
            try:
                # Check if TensorFlow can see the GPU
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Enable memory growth to avoid allocating all GPU memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"TensorFlow is using {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
                    
                    # For AMD GPUs, we need to set some environment variables
                    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                    
                    # Check if ROCm is available (for AMD GPUs)
                    if any('rocm' in gpu.name.lower() for gpu in gpus):
                        print("ROCm detected for AMD GPU acceleration")
                else:
                    print("No GPU found for TensorFlow. Using CPU.")
                    self.use_gpu = False
            except Exception as e:
                print(f"Error initializing TensorFlow GPU: {e}")
                print("Falling back to CPU for TensorFlow operations.")
                self.use_gpu = False
        
        # Enable OpenCL for OpenCV if GPU is available
        if self.use_gpu:
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                print("OpenCL is enabled for OpenCV operations")
            else:
                print("OpenCL could not be enabled for OpenCV. Some operations will use CPU.")
        
        # Initialize the recognizer based on the method
        if method == 'eigenfaces':
            self.recognizer = cv2.face.EigenFaceRecognizer_create()
        elif method == 'lbph':
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        elif method == 'ml':
            # Using SVM for machine learning approach
            self.recognizer = SVC(kernel='linear', probability=True)
            self.label_encoder = LabelEncoder()
        elif method == 'deep':
            # Using a deep learning model for face recognition
            self._create_deep_model()
        else:
            raise ValueError(f"Unsupported recognition method: {method}")
    
    def _create_deep_model(self):
        """Create a deep learning model for face recognition."""
        # Create a simple CNN model for face recognition
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            # Output layer will be added during training when we know the number of classes
        ])
        
        self.deep_model = model
        self.label_encoder = LabelEncoder()
    
    def preprocess_face(self, face):
        """
        Preprocess a face image for recognition.
        
        Args:
            face: Face image
            
        Returns:
            Preprocessed face image
        """
        # Resize to standard size
        face = cv2.resize(face, self.face_size)
        
        # Convert to grayscale if not already
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        face = cv2.equalizeHist(face)
        
        return face
    
    def extract_faces(self, image, face_locations):
        """
        Extract face images from the main image.
        
        Args:
            image: Input image
            face_locations: List of face rectangles (x, y, w, h)
            
        Returns:
            List of extracted and preprocessed face images
        """
        faces = []
        for (x, y, w, h) in face_locations:
            face = image[y:y+h, x:x+w]
            faces.append(self.preprocess_face(face))
        return faces
    
    def train(self, faces, labels):
        """
        Train the face recognizer.
        
        Args:
            faces: List of face images
            labels: List of corresponding labels
        """
        if self.method in ['eigenfaces', 'lbph']:
            self.recognizer.train(faces, np.array(labels))
        
        elif self.method == 'ml':
            # For ML approach, flatten the images
            flattened_faces = [face.flatten() for face in faces]
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.recognizer.fit(flattened_faces, encoded_labels)
        
        elif self.method == 'deep':
            # For deep learning approach
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)
            num_classes = len(self.label_encoder.classes_)
            
            # Add output layer with correct number of classes
            if len(self.deep_model.layers) == 9:  # If output layer not added yet
                self.deep_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            
            # Compile the model
            self.deep_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Prepare data for training
            # Reshape faces for CNN input (add channel dimension)
            faces_array = np.array([face.reshape(self.face_size[0], self.face_size[1], 1) for face in faces])
            
            # Normalize pixel values
            faces_array = faces_array / 255.0
            
            # Train the model
            self.deep_model.fit(
                faces_array, 
                encoded_labels,
                epochs=10,
                batch_size=32,
                validation_split=0.2
            )
    
    def predict(self, face):
        """
        Predict the identity of a face.
        
        Args:
            face: Preprocessed face image
            
        Returns:
            Tuple of (label, confidence)
        """
        face = self.preprocess_face(face)
        
        if self.method in ['eigenfaces', 'lbph']:
            label, confidence = self.recognizer.predict(face)
            return label, confidence
        
        elif self.method == 'ml':
            # For ML approach
            face_flattened = face.flatten().reshape(1, -1)
            label = self.recognizer.predict(face_flattened)[0]
            proba = self.recognizer.predict_proba(face_flattened)[0]
            confidence = proba[label] * 100
            
            # Convert numeric label back to original label
            original_label = self.label_encoder.inverse_transform([label])[0]
            return original_label, confidence
        
        elif self.method == 'deep':
            # For deep learning approach
            # Reshape and normalize the face
            face_array = face.reshape(1, self.face_size[0], self.face_size[1], 1) / 255.0
            
            # Get prediction
            predictions = self.deep_model.predict(face_array)
            label_index = np.argmax(predictions[0])
            confidence = predictions[0][label_index] * 100
            
            # Convert numeric label back to original label
            original_label = self.label_encoder.inverse_transform([label_index])[0]
            return original_label, confidence
    
    def save_model(self, path):
        """Save the trained model to a file."""
        if self.method in ['eigenfaces', 'lbph']:
            self.recognizer.save(path)
        
        elif self.method == 'ml':
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.recognizer,
                    'encoder': self.label_encoder
                }, f)
        
        elif self.method == 'deep':
            # Save the Keras model
            model_path = path.replace('.xml', '.h5')
            self.deep_model.save(model_path)
            
            # Save the label encoder
            encoder_path = path.replace('.xml', '_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
    
    def load_model(self, path):
        """Load a trained model from a file."""
        if self.method in ['eigenfaces', 'lbph']:
            self.recognizer.read(path)
        
        elif self.method == 'ml':
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.recognizer = data['model']
                self.label_encoder = data['encoder']
        
        elif self.method == 'deep':
            # Load the Keras model
            model_path = path.replace('.xml', '.h5')
            if os.path.exists(model_path):
                self.deep_model = tf.keras.models.load_model(model_path)
            else:
                raise FileNotFoundError(f"Deep learning model not found at {model_path}")
            
            # Load the label encoder
            encoder_path = path.replace('.xml', '_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            else:
                raise FileNotFoundError(f"Label encoder not found at {encoder_path}")