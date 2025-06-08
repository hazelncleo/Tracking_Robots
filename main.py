
import cv2
import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt


def apply_robot_tracking(input_video_file : str, 
                        new_video_file = 'tracked.mp4',
                        pixel_length_ratio = 70/1170,
                        image_threshold_value = 160,
                        erode_kernel_size = 9,
                        border = 200,
                        n_frames_animate = 100,
                        alpha = 0.5,
                        n_filter = 15,
                        text_location = (100,100),
                        plot_velocity = False,
                        plot_file = 'Velocity_Plot.png',
                        orientation = False,
                        plot_orientation = False,
                        n_prongs = 3,
                        tracking_size = 100,
                        min_area = 800,
                        max_area = 20000
                        ):
    '''
    -----------------------------------------------------------------------
    This function will apply the robot tracking algorithm to a specified video file
    -----------------------------------------------------------------------
    Inputs
    -----------------------------------------------------------------------
    video_file : str
        The path to the video file that the algorithm will be applied to
    
    new_file : str
        The path that the new video will be saved to
    
    length_ratio : float
        The ratio of mm to pixels in the video (i.e. 70/1170 = 70 millimetres per 1170 pixels)
        
    threshold : int
        The value to be used in thresholding the image. Must be a value between 0 and 255. Where 0 corresponds to only pixels of value 0 becoming black and 255 means the entire image becomes black.
    
    erode : int
        Size of kernel to be used in erosion. This makes a 1x1 black pixel become an erodexerode block. This must be smaller than the image dimensions
        
    border : int
        Size of border around the outside that shouldn't be considered for tracking in pixels. Ensure that the robot to be tracked does not enter this zone to maintain accurate tracking.
        
    n_frames : int
        Number of frames that the drawn line should show
    
    alpha : float 
        The alpha of the drawn line. 0 corresponds to completely transparent and 1 is opaque.
        
    n_filter : int
        The number of frames to be used in the velocity filtering, higher n corresponds to greater smoothing
    
    text_location : tuple, (int,int)
        Pixel coordinate of the bottom left of the text string 

    plot_velocity : boolean
        If set to True, then the velocity over time will be plotted
    
    plot_file : str
        File path to save the velocity plot to. Must include filename.png at end
        
    orientation : bool
        If set to true then the orientation of the robot will be tracked
    
    n_prongs : int
        Number of prongs in the robot
    -----------------------------------------------------------------------
    '''
    
    # Import video file to use for tracking
    video_data = cv2.VideoCapture(input_video_file)
    
    # Video attributes
    fps = video_data.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Initialize video writer object to save the video with tracking added
    video_out = cv2.VideoWriter(new_video_file, fourcc, fps, frame_size)
    
    # Initialize values
    centroids = []
    frames = []
    velocities = []
    
    if orientation:
        orientation_angles = np.empty((n_prongs,frame_count))
    
    # Loop until no frame returned
    for current_frame_number in range(frame_count):
        
        # Read next frame
        ret, frame = video_data.read()
        
        # If no frame to be read break from while loop
        if not ret:
            break
        
        # Save a copy of the frame
        original_frame = frame.copy()
        
        # edit the frame for robot tracking
        final_edited_image = edit_image_for_tracking(frame, image_threshold_value, erode_kernel_size, border)
        
        # calculate and append the centroid of the black pixels
        centroids.append(get_centroid(final_edited_image))
        
        
        # track the orientation of the robot
        if orientation:
            frame = track_orientation(frame, n_prongs, centroids, final_edited_image, tracking_size, frame_size, min_area, max_area, current_frame_number, orientation_angles)
            

        # draw lines showing path of robot over last "n_frames_animate" frames
        draw_lines(frame, centroids, current_frame_number, n_frames_animate)
        
        
                
        # Apply alpha channel to drawn lines
        frame = cv2.addWeighted(frame, alpha, original_frame, 1-alpha, 0)
        
        # Set first velocity to 0, else calculate the velocity based on finite difference
        if current_frame_number == 0:
            velocities.append(0)
        else:
            velocities.append(calculate_velocity(centroids,pixel_length_ratio,fps))
                
                
        
        # Append the frame to the list of frames
        frames.append(frame)
        
    
    
    filtered_velocity = add_velocity_text_to_frames(frames, velocities, video_out, n_filter, text_location)
           
    # Finalize the video
    video_data.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    
    print('-----------------------------------')
    print('Video Tracking Completed for file: ' + input_video_file)
    print('Saved to new file: ' + new_video_file)
    print('-----------------------------------')
    
    
    if plot_orientation:
        plot_orientation_graph(orientation_angles, frame_count)
        
        
    if plot_velocity:
        plot_velocity_graph(filtered_velocity, fps, plot_file)
        





def plot_orientation_graph(orientation_angles, frame_count):
    
    f,ax = plt.subplots(1,1,subplot_kw={'projection': 'polar'})

    #ax.plot(a,np.arange(len(a)),'-k')
    #ax.plot(b,np.arange(len(b)),'-r')
    ax.plot(orientation_angles[0],np.arange(frame_count),'-g')
    ax.plot(orientation_angles[1],np.arange(frame_count),'-k')
    ax.plot(orientation_angles[2],np.arange(frame_count),'-r')
        
    plt.show()
    
def plot_velocity_graph(filtered_velocity, fps, plot_file):
    
    # Initialize figure
    f,ax = plt.subplots(1,1)
    t = np.arange(0,np.size(filtered_velocity)/fps,1/fps)
    ax.plot(t, filtered_velocity, '-k')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (mm/s)')
    f.suptitle('Velocity of Robot')
    
    plt.savefig(plot_file)
    
    print('-----------------------------------')
    print('Velocity plot Saved to: ' + plot_file)
    print('-----------------------------------')

def edit_image_for_tracking(frame,
                            image_threshold_value,
                            erode_kernel_size,
                            border):
    
    # Convert the frame to grayscale for circle detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to the frame
    _,thresh = cv2.threshold(gray_frame, image_threshold_value, 255, cv2.THRESH_BINARY)
    
    # Erode using an [erode,erode] shaped kernel while excluding a region of 'border' thickness around the edge of the frame
    if border == 0:
        final_edited_image = cv2.erode(thresh, np.ones([erode_kernel_size,erode_kernel_size]))
    else:
        eroded = cv2.erode(thresh[border:-border,border:-border], np.ones([erode_kernel_size, erode_kernel_size]))
    
        # Fill border with white
        final_edited_image = 255 + (0*gray_frame)
        final_edited_image[border:-border,border:-border] = eroded
            
    return final_edited_image 




def get_centroid(final_edited_image):
    
    # Get coordinates of each black pixel
    coords = np.column_stack(np.where(final_edited_image == 0))
        
    # Calculate the centroid of all black pixels
    return np.round(np.sum(coords, axis = 0)/np.shape(coords)[0])


def track_orientation(frame,
                      n_prongs,
                      centroids,
                      final_edited_image,
                      tracking_size,
                      frame_size,
                      min_area,
                      max_area,
                      current_frame_number,
                      orientation_angles
                      ):
    
    
    # Create array of angles for this frame
    current_angles = np.empty(n_prongs)
    
    # get the centroid coordinates of the black pixels for the current frame
    [y,x] = [int(val) for val in centroids[-1]]
    
    
    # Save a TRACKING_FRAMExTRACKING_FRAME frame around the centroid (this takes into account the size of the frame)
    tracking_frame = final_edited_image[np.max([y-tracking_size,0]):np.min([y+tracking_size,frame_size[1]]),np.max([x-tracking_size,0]):np.min([x+tracking_size,frame_size[0]])]
    
    # Find all the contours in the tracking frame
    contours,_ = cv2.findContours(tracking_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Get the coordinates of the middle of the frame
    mid_point = (np.array(np.shape(tracking_frame))/2).astype(int)
    
    j = 0
    for contour in contours:
        
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # If the area is within the threshold then use it for orientation tracking
        if (area > min_area) and (area < max_area) and j < n_prongs:
            
            # Calculate the angle of a set of a contour using PCA
            angle,center = getOrientation(contour,tracking_frame)
            
            # Calculate the Angle of the centre of the contour
            center_angle = np.arctan2(center[0]-mid_point[0],mid_point[1]-center[1])
            
            # Calculate the difference between the PCA angle and the center angle
            diff = np.pi - abs(abs(angle - center_angle) - np.pi)

            # If the PCA angle is pointing towards the centre of the image flip it by pi radians
            if diff > 0.8*np.pi:
                if angle < 0:
                    angle = angle + np.pi
                else:
                    angle = angle - np.pi
            
            
            current_angles[j] = angle
            
            j += 1
            
    # If first frame, or, no contour angles recorded then assign 
    if (current_frame_number == 0) or np.linalg.norm(current_angles) < 1e-6:
        orientation_angles[:,current_frame_number] = current_angles

        draw_angled_line(frame, x, y, orientation_angles[0,current_frame_number])
    else:
        temp_angles,temp_current_angles = np.meshgrid(orientation_angles[:,current_frame_number-1], current_angles)
        
        difference = np.pi - np.abs(np.abs(temp_angles-temp_current_angles) - np.pi)
        
        orientation_angles[:,current_frame_number] = current_angles[np.argmin(difference, axis = 0)]
        
        draw_angled_line(frame,x,y,orientation_angles[0,current_frame_number])
        
    return frame

def draw_lines(frame, centroids, current_frame_number, n_frames_animate):
    
    # If the current number of frames is less than n_frames track that many frames, else track last n_frames
    if current_frame_number <= n_frames_animate:
        for coord_1,coord_2 in zip(centroids[:-1],centroids[1:]):
            cv2.line(frame, (int(coord_1[1]), int(coord_1[0])), (int(coord_2[1]), int(coord_2[0])), (0,0,255), 4) 
    else:
        for coord_1,coord_2 in zip(centroids[-n_frames_animate:-1],centroids[-(n_frames_animate+1):]):
            cv2.line(frame, (int(coord_1[1]), int(coord_1[0])), (int(coord_2[1]), int(coord_2[0])), (0,0,255), 4) 

        
def draw_angled_line(frame, x, y, angle):

    
    y_start = int(y+0*np.cos(angle))
    x_start = int(x-0*np.sin(angle))
    y_end = int(y-25*np.cos(angle))
    x_end = int(x+25*np.sin(angle))
    
    cv2.line(frame, (x_start,y_start), (x_end,y_end), (0,0,0), 6)
    
def calculate_velocity(centroids, pixel_length_ratio, fps):
    # calculate velocity using finite differencing of position
    return np.sqrt(np.power((centroids[-1][1] - centroids[-2][1])*pixel_length_ratio*fps, 2) + np.power((centroids[-1][0] - centroids[-2][0])*pixel_length_ratio*fps, 2))    
        
def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]

    angle = np.arctan2(eigenvectors[1,1], eigenvectors[1,0]) # orientation in radians
    ## [visualization]
    
    return angle, cntr
        
def test_thresholds(video_file, thresholds, border):
    # Import video file
    vid = cv2.VideoCapture(video_file)
    
    # Get frames per second
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    # Video writing to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec
    frame_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))   
    
        
    # Read next frame
    ret, frame = vid.read()
    
    # Save a copy of the frame
    original = frame.copy()
    
    # Convert the frame to grayscale for circle detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    for threshold in thresholds:
        # Apply thresholding to the frame
        _,thresh = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        
        # Fill border with white
        
        
        erode = cv2.erode(thresh, np.ones([20,20]))
        
        final = 255 + (0*gray_frame)
        final[border:-border,border:-border] = erode[border:-border,border:-border]
        
        cv2.imwrite('thresh_{}.png'.format(threshold), final)
        
def add_velocity_text_to_frames(frames, velocities, video_out, n_filter, text_location):
    
    # Filter the velocity to reduce noise
    filtered_velocity = lfilter([1/n_filter]*n_filter,1,velocities)   
    
    # Add the text to the frame and write the frame to the video file
    for filtered,frame in zip(filtered_velocity,frames):
        cv2.putText(frame,'Velocity = {:.2f} mm/s'.format(filtered),
                    text_location,
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
                    color = (0,0,0),
                    thickness = 2)
        
        video_out.write(frame)
        
    return filtered_velocity
        
def main():
    
    if False:
        apply_robot_tracking('video.mov', plot_velocity = True, new_file = 'tracked_orient.mp4', orientation = True, n_prongs = 2)
    
    if False:
        test_thresholds('Robotic_Movement.mp4', thresholds = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250], border = 50)
    
    if True:
        apply_robot_tracking('3Prong_1.mp4',
                            plot_velocity=False,
                            new_video_file='tracked_3Prong_1.mp4',
                            border=50,
                            image_threshold_value=140,
                            erode_kernel_size=18,
                            pixel_length_ratio=75/1056,
                            n_frames_animate=15,
                            orientation=True
                            )
    
    if False:
        frame = cv2.imread('test.png')
        
        draw_line(frame,100,100,0)
        draw_line(frame,100,100,np.pi/2)
        draw_line(frame,100,100,np.pi)
        draw_line(frame,100,100,3*np.pi/2)
        
        cv2.imwrite('test_1.png',frame)
    
    if False:
        frame = cv2.imread('test.png')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours,_ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        middle = np.array([int(frame.shape[0]/2), int(frame.shape[1]/2)])
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if (area > 1000) and (area < 10000):
                
                angle, center = getOrientation(contour,frame)
                
                if (angle < 0) and (angle > -np.pi/2):
                    # top left
                    quad_theta = 1
                elif angle < -np.pi/2:
                    # bottom left
                    quad_theta = 0
                elif (angle > 0) and (angle < np.pi/2):
                    # top right
                    quad_theta = 2
                else:
                    # bottom right    
                    quad_theta = 3
                    
                dx = center - middle
                
                if (dx[0] < 0) and (dx[1] < 0):
                    # top left
                    quad_dx = 1
                elif (dx[0] > 0) and (dx[1] < 0):
                    # top right
                    quad_dx = 2
                elif (dx[0] < 0) and (dx[1] > 0):
                    # bottom left
                    quad_dx = 0
                else:
                    # bottom right
                    quad_dx = 3
                
                if abs(quad_dx - quad_theta) == 2:
                    if angle < 0:
                        angle = angle + np.pi
                    else:
                        angle = angle - np.pi
                    
                
                draw_line(frame, center[0], center[1], angle)
                
        

        cv2.imwrite('bruh_2.png',frame)




if __name__ == '__main__':
    main()