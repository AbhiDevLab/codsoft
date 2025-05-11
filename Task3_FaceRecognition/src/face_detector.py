import cv2
import numpy as np
import pyopencl as cl
import os

class FaceDetector:
    """Class for detecting faces in images using different methods with GPU acceleration."""
    
    def __init__(self, method='haar', model_path=None, use_gpu=True):
        """
        Initialize the face detector.
        
        Args:
            method (str): Detection method ('haar', 'dnn', or 'dnn_gpu')
            model_path (str): Path to the model file
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.method = method
        self.use_gpu = use_gpu
        
        # Initialize OpenCL context and queue if GPU is to be used
        if self.use_gpu:
            try:
                # Get OpenCL platforms
                platforms = cl.get_platforms()
                if not platforms:
                    print("No OpenCL platforms found. Falling back to CPU.")
                    self.use_gpu = False
                else:
                    # Try to get AMD platform
                    amd_platform = None
                    for platform in platforms:
                        if 'AMD' in platform.name or 'Radeon' in platform.name:
                            amd_platform = platform
                            break
                    
                    if amd_platform:
                        # Get GPU devices from AMD platform
                        gpu_devices = amd_platform.get_devices(device_type=cl.device_type.GPU)
                        if gpu_devices:
                            # Create context and command queue
                            self.cl_context = cl.Context(devices=gpu_devices)
                            self.cl_queue = cl.CommandQueue(self.cl_context)
                            print(f"Using AMD GPU: {gpu_devices[0].name}")
                        else:
                            print("No AMD GPU devices found. Falling back to CPU.")
                            self.use_gpu = False
                    else:
                        # If no AMD platform, try to use any available GPU
                        for platform in platforms:
                            gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                            if gpu_devices:
                                self.cl_context = cl.Context(devices=gpu_devices)
                                self.cl_queue = cl.CommandQueue(self.cl_context)
                                print(f"Using GPU: {gpu_devices[0].name} from {platform.name}")
                                break
                        else:
                            print("No GPU devices found. Falling back to CPU.")
                            self.use_gpu = False
            except Exception as e:
                print(f"Error initializing GPU: {e}")
                print("Falling back to CPU.")
                self.use_gpu = False
        
        # Initialize the detector based on the method
        if method == 'haar':
            if model_path is None:
                model_path = 'models/haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(model_path)
            
            # Set OpenCL flag for Haar Cascade if GPU is available
            if self.use_gpu:
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    print("OpenCL is enabled for Haar Cascade detection")
                else:
                    print("OpenCL could not be enabled. Using CPU for Haar Cascade.")
        
        elif method == 'dnn' or method == 'dnn_gpu':
            # Load DNN face detector
            prototxt_path = 'models/deploy.prototxt'
            model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
            
            if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
                print(f"DNN model files not found. Downloading...")
                self._download_dnn_model(prototxt_path, model_path)
            
            self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            # Enable OpenCL backend for DNN if GPU is available and method is dnn_gpu
            if method == 'dnn_gpu' and self.use_gpu:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCL)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                print("Using OpenCL backend for DNN")
        else:
            raise ValueError(f"Unsupported detection method: {method}")
    
    def _download_dnn_model(self, prototxt_path, model_path):
        """Download the DNN model files if they don't exist."""
        import urllib.request
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(prototxt_path), exist_ok=True)
        
        # Download prototxt
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        
        # Download model
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        urllib.request.urlretrieve(model_url, model_path)
        
        print("DNN model files downloaded successfully")
    
    def detect_faces(self, image, min_confidence=0.5):
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            min_confidence: Minimum confidence threshold for DNN detection
            
        Returns:
            List of face rectangles as (x, y, w, h)
        """
        # Convert image to UMat for OpenCL acceleration if GPU is enabled
        if self.use_gpu and self.method == 'haar':
            try:
                # Convert to UMat for GPU processing
                umat_image = cv2.UMat(image)
                gray = cv2.cvtColor(umat_image, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                # Convert back to CPU for further processing
                if isinstance(faces, cv2.UMat):
                    faces = faces.get()
                return faces
            except Exception as e:
                print(f"GPU detection failed: {e}. Falling back to CPU.")
                # Fall back to CPU if GPU detection fails
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                return faces
        elif self.method == 'haar':
            # CPU-based Haar cascade detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        
        # Use DNN for face detection (with or without GPU)
        elif self.method in ['dnn', 'dnn_gpu']:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 
                1.0, (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > min_confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Convert to (x, y, w, h) format
                    faces.append((startX, startY, endX - startX, endY - startY))
            
            return faces
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles around detected faces.
        
        Args:
            image: Input image
            faces: List of face rectangles (x, y, w, h)
            color: Rectangle color
            thickness: Line thickness
            
        Returns:
            Image with drawn rectangles
        """
        img_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        return img_copy
