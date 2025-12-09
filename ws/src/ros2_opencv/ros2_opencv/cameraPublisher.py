# Import the libraries
# this is OpenCV
import cv2
import os
import time

# these are ROS2 package modules and libraries
import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node 
from cv_bridge import CvBridge
 
# the argument "Node" means that the PublisherNodeClass inherits (or is a child of)
# the class called Node. The Node class is a standard ROS2 class
class PublisherNodeClass(Node):
    

    # constructor   
    def __init__(self, watch_dir="/tmp/frames"):

        # this function is used to initialize the attributes of the parent class 
        super().__init__('publisher_node')
      
        # here, we create an instance of the OpenCV VideoCapture object
        # this is the camera device number - you need to properly adjust this number
        # depending on the camera device number assignmed by the Linux system
        # self.cameraDeviceNumber=0
        # self.camera = cv2.VideoCapture(self.cameraDeviceNumber)

        # Start cam process
        #self.cam_process = subprocess.Popen(
        #        ["cam", "-c", "1", "-C"],
        #        stdout=subprocess.PIPE,
        #        bufsize=10**8
        #)

        # Directory where cam saves frames
        self.watch_dir = watch_dir
        os.makedirs(self.watch_dir, exist_ok=True)
         
        # CvBridge is used to convert OpenCV images to ROS2 messages that can be sent throught the topics
        self.bridgeObject = CvBridge()
    
        # name of the topic used to transfer the camera images
        # this topic name should match the topic name in the subscriber node
        self.topicNameFrames='topic_camera_image'
 
        # the queue size for messages
        self.queueSize=20
    
        # here, the function "self.create_publisher" creates the publisher that
        # publishes the messages of the type Image, over the topic self.topicNameFrames
        # and with self.queueSize
        self.publisher = self.create_publisher(Image, self.topicNameFrames, self.queueSize)
      
        # communication period in seconds
        self.periodCommunication = 0.1  
    
        # Create the timer that calls the function self.timer_callback every self.periodCommunication seconds
        self.timer = self.create_timer(self.periodCommunication, self.timer_callbackFunction)
        self.last_seen = set()

        # this is the counter tracking how many images are published
        self.i = 0
    
    # this is the callback function that is called every self.periodCommunication seconds
    def timer_callbackFunction(self):
    
        # here we read the frame by using the camera
        # success, frame = self.camera.read()
        # resize the image
        # frame = cv2.resize(frame, (820,640), interpolation=cv2.INTER_CUBIC) 
        
        # Read raw bytes from cam
        #raw_bytes = self.cam_process.stdout.read(1024*1024) #adjust buffer size
        #if not raw_bytes:
        #    return

        # Decode JPEG frame
        #np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        #frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # if we are able to read the frame
        #if frame is not None:
            # here, we convert OpenCV frame to
            # ROS2ImageMessage=self.bridgeObject.cv2_to_imgmsg(frame)
            #ROS2ImageMessage = self.bridgeObject.cv2_to_imgmsg(frame, encoding="bgr8")
            # publish the image
            #self.publisher.publish(ROS2ImageMessage)
            # Use the logger to display the message on the screen
            #self.get_logger().info('Publishing image number %d' % self.i)
            # update the image counter
            #self.i += 1
        
        # List files in directory
        files = sorted(os.listdir(self.watch_dir))
        for f in files:
            path = os.path.join(self.watch_dir, f)
            if path not in self.last_seen and os.path.isfile(path):
                # Load image with OpenCV
                frame = cv2.imread(path)
                if frame is None:
                    continue

                # Convert to ROS2 Image message
                msg = self.bridgeObject.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher.publish(msg)

                self.get_logger().info(f'Published image {self.i} from {path}')
                self.i += 1
                self.last_seen.add(path)

# this is the main function and this is the entry point of our code
def main(args=None):
  # initialize rclpy 
  rclpy.init(args=args)
  
  # create the publisher object
  publisherObject = PublisherNodeClass("captures/")
  
  # here we spin, and the callback timer function is called recursively
  rclpy.spin(publisherObject)
  
  # destroy
  publisherObject.destroy_node()
  
  # Shutdown
  rclpy.shutdown()
  
if __name__ == '__main__':
    main()
