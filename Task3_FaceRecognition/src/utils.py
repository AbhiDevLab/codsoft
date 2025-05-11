import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    """
    Load all images from a folder and its subfolders.
    
    Args:
        folder: Path to the folder
        
    Returns:
        Tuple of (images, labels)
    """
    images = []
    labels = []
    
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        
        if not os.path.isdir(person_folder):
            continue
            
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            
            if not os.path.isfile(img_path):
                continue
                
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(person_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, labels

def display_image(image, title=None, figsize=(10, 8)):
    """
    Display an image using matplotlib.
    
    Args:
        image: Image to display
        title: Optional title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Convert BGR to RGB for display
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def benchmark_gpu():
    """Benchmark GPU performance for OpenCL and TensorFlow."""
    results = {}
    
    # Test OpenCL performance
    try:
        import pyopencl as cl
        import time
        
        # Get platforms
        platforms = cl.get_platforms()
        if platforms:
            for platform in platforms:
                print(f"Testing platform: {platform.name}")
                
                # Get devices
                devices = platform.get_devices()
                for device in devices:
                    if device.type == cl.device_type.GPU:
                        print(f"  Testing GPU: {device.name}")
                        
                        # Create context and queue
                        ctx = cl.Context([device])
                        queue = cl.CommandQueue(ctx)
                        
                        # Create test data
                        data_size = 10000000  # 10M elements
                        a = np.random.rand(data_size).astype(np.float32)
                        b = np.random.rand(data_size).astype(np.float32)
                        
                        # Create buffers
                        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                        b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                        c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, a.nbytes)
                        
                        # Create program
                        program = cl.Program(ctx, """
                        __kernel void add(__global const float *a, __global const float *b, __global float *c) {
                            int gid = get_global_id(0);
                            c[gid] = a[gid] + b[gid];
                        }
                        """).build()
                        
                        # Run kernel
                        start_time = time.time()
                        event = program.add(queue, (data_size,), None, a_buf, b_buf, c_buf)
                        event.wait()
                        
                        # Get result
                        c = np.empty_like(a)
                        cl.enqueue_copy(queue, c, c_buf)
                        
                        end_time = time.time()
                        
                        results[device.name] = {
                            'type': 'OpenCL',
                            'time': end_time - start_time,
                            'data_size': data_size
                        }
                        
                        print(f"    OpenCL test completed in {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"OpenCL benchmark failed: {e}")
    
    # Test TensorFlow performance
    try:
        import tensorflow as tf
        import time
        
        # Check if TensorFlow can see the GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Create test data
            data_size = 1000  # Matrix size
            a = tf.random.normal((data_size, data_size))
            b = tf.random.normal((data_size, data_size))
            
            # Warm up
            c = tf.matmul(a, b)
            
            # Run test
            start_time = time.time()
            c = tf.matmul(a, b)
            # Force execution
            c_val = c.numpy()
            end_time = time.time()
            
            for gpu in gpus:
                results[gpu.name.decode('utf-8')] = {
                    'type': 'TensorFlow',
                    'time': end_time - start_time,
                    'data_size': data_size
                }
                
                print(f"TensorFlow test on {gpu.name.decode('utf-8')} completed in {end_time - start_time:.4f} seconds")
        else:
            print("TensorFlow cannot access any GPUs")
    except Exception as e:
        print(f"TensorFlow benchmark failed: {e}")
    
    return results
