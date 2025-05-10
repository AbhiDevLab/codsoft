import os
import cv2
import numpy as np
import argparse
import time
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.utils import load_images_from_folder, display_image, create_directory

def setup_project_structure():
    """Create the necessary directories for the project."""
    directories = [
        'data/known_faces',
        'data/test_images',
        'models'
    ]
    
    for directory in directories:
        create_directory(directory)
    
    # Download Haar cascade if it doesn't exist
    haar_path = 'models/haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_path):
        print("Downloading Haar cascade model...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        import urllib.request
        urllib.request.urlretrieve(url, haar_path)
        print(f"Downloaded to {haar_path}")

def check_gpu_support():
    """Check if GPU acceleration is available."""
    # Check OpenCL support (for AMD GPUs)
    opencl_available = cv2.ocl.haveOpenCL()
    if opencl_available:
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.useOpenCL():
            print("OpenCL is available and enabled for OpenCV")
            
            # Get OpenCL devices
            import pyopencl as cl
            try:
                platforms = cl.get_platforms()
                for platform in platforms:
                    print(f"Platform: {platform.name}")
                    devices = platform.get_devices()
                    for device in devices:
                        if device.type == cl.device_type.GPU:
                            print(f"  GPU Device: {device.name}")
                            return True
            except Exception as e:
                print(f"Error getting OpenCL devices: {e}")
        else:
            print("OpenCL is available but could not be enabled")
    else:
        print("OpenCL is not available")
    
    # Check TensorFlow GPU support
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow can access {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  {gpu.name}")
            return True
        else:
            print("TensorFlow cannot access any GPUs")
    except Exception as e:
        print(f"Error checking TensorFlow GPU support: {e}")
    
    return False

def train_face_recognition(data_dir='data/known_faces', method='eigenfaces', use_gpu=True):
    """Train the face recognition model."""
    # Initialize face detector and recognizer
    detector = FaceDetector(method='dnn_gpu' if use_gpu else 'haar', use_gpu=use_gpu)
    recognizer = FaceRecognizer(method=method, use_gpu=use_gpu)
    
    # Load training images
    print(f"Loading training images from {data_dir}...")
    images, labels = load_images_from_folder(data_dir)
    
    if not images:
        print("No training images found. Please add images to the data/known_faces directory.")
        return None
    
    print(f"Loaded {len(images)} images with {len(set(labels))} unique labels.")
    
    # Extract faces from images
    faces = []
    valid_labels = []
    
    start_time = time.time()
    for img, label in zip(images, labels):
        detected_faces = detector.detect_faces(img)
        
        if len(detected_faces) == 1:  # Only use images with exactly one face
            face_img = recognizer.extract_faces(img, detected_faces)[0]
            faces.append(face_img)
            valid_labels.append(label)
    
    detection_time = time.time() - start_time
    print(f"Face detection completed in {detection_time:.2f} seconds")
    
    if not faces:
        print("No faces detected in training images.")
        return None
    
    print(f"Extracted {len(faces)} faces for training.")
    
    # Train the recognizer
    print("Training face recognizer...")
    start_time = time.time()
    recognizer.train(faces, valid_labels)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the model
    model_path = f'models/face_recognizer_{method}.xml'
    recognizer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return recognizer

def recognize_faces_in_image(image_path, detector, recognizer):
    """Recognize faces in an image."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Detect faces
    start_time = time.time()
    faces = detector.detect_faces(image)
    detection_time = time.time() - start_time
    print(f"Detected {len(faces)} faces in {detection_time:.2f} seconds.")
    
    # Draw rectangles and labels
    result_image = image.copy()
    
    start_time = time.time()
    for (x, y, w, h) in faces:
        # Extract the face
        face = image[y:y+h, x:x+w]
        
        # Recognize the face
        label, confidence = recognizer.predict(face)
        
        # Draw rectangle
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw label
        text = f"{label} ({confidence:.2f}%)"
        cv2.putText(result_image, text, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    recognition_time = time.time() - start_time
    print(f"Recognition completed in {recognition_time:.2f} seconds.")
    
    return result_image

def process_video(video_path, detector, recognizer, output_path=None, display=True):
    """Process a video for face recognition."""
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Performance metrics
    frame_count = 0
    total_detection_time = 0
    total_recognition_time = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        detection_start = time.time()
        faces = detector.detect_faces(frame)
        detection_time = time.time() - detection_start
        total_detection_time += detection_time
        
        # Draw rectangles and labels
        recognition_start = time.time()
        for (x, y, w, h) in faces:
            # Extract the face
            face = frame[y:y+h, x:x+w]
            
            # Recognize the face
            try:
                label, confidence = recognizer.predict(face)
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw label
                text = f"{label} ({confidence:.2f}%)"
                cv2.putText(frame, text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error recognizing face: {e}")
        
        recognition_time = time.time() - recognition_start
        total_recognition_time += recognition_time
        
        # Add FPS info to frame
        current_fps = 1.0 / (detection_time + recognition_time) if (detection_time + recognition_time) > 0 else 0
        cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Write the frame
        if output_path:
            out.write(frame)
        
        # Display the frame
        if display:
            cv2.imshow('Face Recognition', frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Calculate and display performance metrics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_detection_time = total_detection_time / frame_count if frame_count > 0 else 0
    avg_recognition_time = total_recognition_time / frame_count if frame_count > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average detection time per frame: {avg_detection_time*1000:.2f} ms")
    print(f"Average recognition time per frame: {avg_recognition_time*1000:.2f} ms")
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Face Detection and Recognition with GPU Acceleration')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['setup', 'check_gpu', 'train', 'image', 'video', 'webcam', 'benchmark'],
                        help='Operation mode')
    parser.add_argument('--method', type=str, default='eigenfaces',
                        choices=['eigenfaces', 'lbph', 'ml', 'deep'],
                        help='Face recognition method')
    parser.add_argument('--detector', type=str, default='dnn_gpu',
                        choices=['haar', 'dnn', 'dnn_gpu'],
                        help='Face detection method')
    parser.add_argument('--input', type=str, help='Input image or video path')
    parser.add_argument('--output', type=str, help='Output path for processed video')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    use_gpu = not args.no_gpu
    
    if args.mode == 'setup':
        setup_project_structure()
        print("Project structure set up successfully.")
    
    elif args.mode == 'check_gpu':
        gpu_available = check_gpu_support()
        if gpu_available:
            print("\nGPU acceleration is available and can be used.")
        else:
            print("\nGPU acceleration is not available. The application will run on CPU.")
    
    elif args.mode == 'train':
        recognizer = train_face_recognition(method=args.method, use_gpu=use_gpu)
        if recognizer:
            print("Training completed successfully.")
    
    elif args.mode == 'image':
        if not args.input:
            print("Please provide an input image path with --input.")
            return
        
        # Load models
        detector = FaceDetector(method=args.detector, use_gpu=use_gpu)
        recognizer = FaceRecognizer(method=args.method, use_gpu=use_gpu)
        
        model_path = f'models/face_recognizer_{args.method}.xml'
        if not os.path.exists(model_path) and args.method != 'deep':
            print(f"Model not found: {model_path}. Please train the model first.")
            return
        elif args.method == 'deep' and not os.path.exists(model_path.replace('.xml', '.h5')):
            print(f"Deep learning model not found. Please train the model first.")
            return
        
        recognizer.load_model(model_path)
        
        # Process the image
        result = recognize_faces_in_image(args.input, detector, recognizer)
        
        if result is not None:
            # Save the result if output path is provided
            if args.output:
                cv2.imwrite(args.output, result)
                print(f"Result saved to {args.output}")
            
            # Display the result
            display_image(result, title="Face Recognition Result")
    
    elif args.mode == 'video':
        if not args.input:
            print("Please provide an input video path with --input.")
            return
        
        # Load models
        detector = FaceDetector(method=args.detector, use_gpu=use_gpu)
        recognizer = FaceRecognizer(method=args.method, use_gpu=use_gpu)
        
        model_path = f'models/face_recognizer_{args.method}.xml'
        if not os.path.exists(model_path) and args.method != 'deep':
            print(f"Model not found: {model_path}. Please train the model first.")
            return
        elif args.method == 'deep' and not os.path.exists(model_path.replace('.xml', '.h5')):
            print(f"Deep learning model not found. Please train the model first.")
            return
        
        recognizer.load_model(model_path)
        
        # Process the video
        process_video(args.input, detector, recognizer, args.output)
    
    elif args.mode == 'webcam':
        # Load models
        detector = FaceDetector(method=args.detector, use_gpu=use_gpu)
        recognizer = FaceRecognizer(method=args.method, use_gpu=use_gpu)
        
        model_path = f'models/face_recognizer_{args.method}.xml'
        if not os.path.exists(model_path) and args.method != 'deep':
            print(f"Model not found: {model_path}. Please train the model first.")
            return
        elif args.method == 'deep' and not os.path.exists(model_path.replace('.xml', '.h5')):
            print(f"Deep learning model not found. Please train the model first.")
            return
        
        recognizer.load_model(model_path)
        
        # Process webcam feed
        process_video(0, detector, recognizer, args.output)
    
    elif args.mode == 'benchmark':
        # Run performance benchmark
        print("Running performance benchmark...")
        
        # Check GPU availability
        gpu_available = check_gpu_support()
        
        # Test different detection methods
        detection_methods = ['haar', 'dnn', 'dnn_gpu'] if gpu_available else ['haar', 'dnn']
        recognition_methods = ['eigenfaces', 'lbph', 'ml', 'deep'] if gpu_available else ['eigenfaces', 'lbph', 'ml']
        
        # Benchmark detection methods
        print("\nBenchmarking face detection methods:")
        for method in detection_methods:
            detector = FaceDetector(method=method, use_gpu=use_gpu)
            
            # Test on a sample image
            if args.input:
                image = cv2.imread(args.input)
                if image is not None:
                    start_time = time.time()
                    faces = detector.detect_faces(image)
                    detection_time = time.time() - start_time
                    print(f"Method: {method}, Detected faces: {len(faces)}, Time: {detection_time:.4f} seconds")
        
        # Benchmark recognition methods (if models are available)
        print("\nBenchmarking face recognition methods:")
        for method in recognition_methods:
            model_path = f'models/face_recognizer_{method}.xml'
            if os.path.exists(model_path) or (method == 'deep' and os.path.exists(model_path.replace('.xml', '.h5'))):
                recognizer = FaceRecognizer(method=method, use_gpu=use_gpu)
                recognizer.load_model(model_path)
                
                # Test on a sample image with detected face
                if args.input:
                    image = cv2.imread(args.input)
                    if image is not None:
                        detector = FaceDetector(method='dnn_gpu' if gpu_available else 'dnn')
                        faces = detector.detect_faces(image)
                        
                        if faces:
                            face = image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
                            
                            start_time = time.time()
                            label, confidence = recognizer.predict(face)
                            recognition_time = time.time() - start_time
                            
                            print(f"Method: {method}, Recognized as: {label}, Confidence: {confidence:.2f}%, Time: {recognition_time:.4f} seconds")
            else:
                print(f"Method: {method}, Model not found. Skipping benchmark.")

if __name__ == "__main__":
    main()