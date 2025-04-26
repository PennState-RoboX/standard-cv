import enum
import math
import pathlib
from dataclasses import dataclass
import cv2
import numpy as np
import serial
from UART_UTIL import send_data, get_imu
from camera_source import CameraSource
from kinematic_prediction import poly_predict
import argparse
import logging
import time
from camera_params import camera_params, DepthSource
from Target import Target
import struct

active_cam_config = None
frame_aligner = None
num = 0  # Add global variable for frame counting


def nothing(x):
    pass


class TargetColor(enum.Enum):
    RED = 'red'
    BLUE = 'blue'


class CVParams:
    def __init__(self, target_color: TargetColor):
        self.target_color = target_color
        if target_color == TargetColor.RED:
            self.hue_min, self.hue_min_range = 4, (0, 180, 1)
            self.hue_max, self.hue_max_range = 38, (0, 180, 1)
            self.saturation_min, self.saturation_min_range = 54, (0, 255, 1)
            self.value_min, self.value_min_range = 111, (0, 255, 1)

            self.close_size = 1
            self.erode_size = 1
            self.dilate_size = 5
        else:
            self.hue_min, self.hue_min_range = 90, (0, 180, 1)
            self.hue_max, self.hue_max_range = 120, (0, 180, 1)
            self.saturation_min, self.saturation_min_range = 20, (0, 255, 1)
            self.value_min, self.value_min_range = 128, (0, 255, 1)

            self.close_size = 3
            self.erode_size = 2
            self.dilate_size = 2

        self.close_size_range = self.erode_size_range = self.dilate_size_range = (
            1, 20, 1)

        self.bar_aspect_ratio_min = 1.1
        self.bar_aspect_ratio_max = 13.0
        self.bar_z_angle_max = 20.0
        self.relative_x_delta_max = 3.0
        self.relative_y_delta_max = 3.0
        self.relative_height_diff_max = 0.5
        self.z_delta_max = 10.0


def createTrackbarsForParams(window_name: str, params: CVParams):
    for key, value in params.__dict__.items():
        if not key.endswith('_range') and type(value) in [int, float]:
            if hasattr(params, key + '_range'):
                slider_min, slider_max, scaling = getattr(
                    params, key + '_range')
            else:
                slider_min = 10 ** math.floor(math.log10(value))
                slider_max = 10 * slider_min
                scaling = 0.01

            cv2.createTrackbar(key, window_name, int(
                slider_min / scaling), int(slider_max / scaling), nothing)
            cv2.setTrackbarPos(key, window_name, int(value / scaling))


def updateParamsFromTrackbars(window_name: str, params: CVParams):
    for key, value in params.__dict__.items():
        if not key.endswith('_range') and type(value) in [int, float]:
            if hasattr(params, key + '_range'):
                scaling = getattr(params, key + '_range')[2]
            else:
                scaling = 0.01

            setattr(params, key, cv2.getTrackbarPos(
                key, window_name) * scaling)


def open_binary(binary, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    dst = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return dst


def close_binary(binary, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    dst = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return dst


def erode_binary(binary, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    dst = cv2.erode(binary, kernel)
    return dst


def dilate_binary(binary, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    dst = cv2.dilate(binary, kernel)
    return dst


# read cap and morphological operation to get led binary image.
def read_morphology(cap, config: CVParams):
    try:
        frame = cap
        if frame is None:
            logger.error("Invalid frame in read_morphology")
            return np.zeros((480, 640), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8)
            
        # Convert to HSV once - this is expensive, so avoid if possible
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Avoid using cv2.split which is slow - use direct numpy indexing instead
        # H, S, V = cv2.split(hsv_image)  # Split channels
        H = hsv_image[:, :, 0]
        S = hsv_image[:, :, 1]
        V = hsv_image[:, :, 2]
    
        # Use vectorized numpy operations - much faster than multiple comparisons
        # Pre-compute and reuse mask operations
        h_mask = (H >= config.hue_min) & (H <= config.hue_max)
        s_mask = (S >= config.saturation_min)
        v_mask = (V >= config.value_min)
        mask_processed = h_mask & s_mask & v_mask
        mask_processed = mask_processed.astype(np.uint8) * 255
    
        # Only create kernels once per frame
        # Define kernels once to avoid recreating them for each operation
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                               (config.close_size, config.close_size))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                               (config.erode_size, config.erode_size))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                (config.dilate_size, config.dilate_size))
    
        # Apply morphological operations
        dst_close = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel_close)
        dst_erode = cv2.erode(dst_close, kernel_erode)
        dst_dilate = cv2.dilate(dst_erode, kernel_dilate)
    
        if debug:
            """
            Display the final image after preprocessing
            """
            cv2.imshow("erode", dst_dilate)
    
        return dst_dilate, frame
    except Exception as e:
        logger.error(f"Error in read_morphology: {e}")
        # Return empty images if processing fails
        empty_binary = np.zeros((480, 640), dtype=np.uint8)
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame.copy()
        return empty_binary, empty_frame


def spherical_to_cartesian(yaw: float, pitch: float, depth: float):
    # Convert once to avoid multiple conversions
    phi_rad = np.radians(90.0 - pitch)
    theta_rad = np.radians(yaw)
    
    # Pre-compute sine and cosine which are expensive operations
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    
    # Use pre-computed values
    return depth * np.array([sin_phi * cos_theta, sin_phi * sin_theta, cos_phi])


def cartesian_to_spherical(coords: np.ndarray):
    # Calculate values once to avoid redundant calculations
    x, y, z = coords
    xy_squared_sum = x * x + y * y
    xy_norm = np.sqrt(xy_squared_sum)
    
    # Calculate angles more efficiently
    yaw = np.rad2deg(np.arctan2(y, x))
    pitch = 90.0 - np.rad2deg(np.arctan2(xy_norm, z))
    
    # Avoid redundant norm calculation
    depth = np.sqrt(xy_squared_sum + z * z)
    
    return yaw, pitch, depth

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def get_3d_target_location(imgPoints, frame, depth_frame):
    try:
        # Cache active_cam_config values to avoid repeated lookups
        cam_matrix = np.array(active_cam_config['camera_matrix'], dtype=np.float64)
        dist_coeffs = np.array(active_cam_config['distort_coeffs'], dtype=np.float64)
        cx, cy = active_cam_config['cx'], active_cam_config['cy']
        fx, fy = active_cam_config['fx'], active_cam_config['fy']
        depth_source = active_cam_config['depth_source']
    
        # Undistort the given image points
        imgPoints = cv2.undistortPoints(
            imgPoints, cam_matrix, dist_coeffs, P=cam_matrix)[:, 0, :]
    
        # Calculate the average (center) point of the image points - use numpy's mean which is faster
        center_point = np.mean(imgPoints, axis=0)
    
        # Calculate the offset of the center point from the camera's optical center
        center_offset = center_point - np.array([cx, cy])
        center_offset[1] = -center_offset[1]
    
        # Convert the offset to angular measurements (yaw and pitch) in degrees - use precomputed fx/fy
        angles = np.rad2deg(np.arctan2(center_offset, np.array([fx, fy])))
        
        # Initialize Yaw and Pitch with default values from angles
        Yaw = angles[0]
        Pitch = angles[1]
        meanDVal = 1000.0  # Default depth value if calculation fails
    
        # Calculate depth based on the configured depth source
        if depth_source == DepthSource.PNP:
            try:
                # Pre-define object points once - avoiding recreating them every time
                width_size_half = 70  # half width of the object
                height_size_half = 62.5  # half height of the object
                objPoints = np.array([[-width_size_half, -height_size_half, 0],
                                      [width_size_half, -height_size_half, 0],
                                      [width_size_half, height_size_half, 0],
                                      [-width_size_half, height_size_half, 0]], dtype=np.float64)
    
                # Use solvePnP_IPPE method to find the object's pose
                retval, rvec, tvec = cv2.solvePnP(
                    objPoints, imgPoints, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                
                # Calculate depth and angles more efficiently
                meanDVal = np.linalg.norm(tvec[:, 0])
                
                # Pre-compute the division factor
                pi_factor = 2 * np.pi
                
                offsetY = 1  # offset for Yaw
                Yaw = np.arctan(tvec[(0,0)]/ tvec[(2,0)]) / pi_factor * 360 - offsetY
                
                offsetP = -4  # offset for Pitch
                Pitch = -(np.arctan(tvec[(1, 0)] / tvec[(2, 0)]) / pi_factor * 360) - offsetP
            except Exception as e:
                logger.error(f"Error in PnP depth calculation: {e}")
                # Keep default values if PnP fails
    
        elif depth_source == DepthSource.STEREO:
            try:
                # Ensure the depth frame is available for stereo depth calculation
                if depth_frame is not None:
                    # Create mask efficiently
                    panel_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    # Use fillPoly directly which is faster than drawContours for this case
                    cv2.fillPoly(panel_mask, [imgPoints.astype(np.int64)], 1)
                    
                    # Only resize if dimensions differ
                    if panel_mask.shape != depth_frame.shape:
                        panel_mask_scaled = cv2.resize(
                            panel_mask, (depth_frame.shape[1], depth_frame.shape[0]))
                    else:
                        panel_mask_scaled = panel_mask
    
                    # Calculate the mean depth value within the masked area
                    meanDVal, _ = cv2.meanStdDev(depth_frame, mask=panel_mask_scaled)
            except Exception as e:
                logger.error(f"Error in stereo depth calculation: {e}")
                # Keep default values if stereo depth fails
        else:
            # Log a warning if an invalid depth source is configured
            logger.warning('Invalid depth source in camera config, using default depth')
    
        # Store and return the calculated depth, yaw, pitch, and image points
        target_Dict = {"depth": meanDVal,
                       "Yaw": Yaw, "Pitch": Pitch, "imgPoints": imgPoints}
        return target_Dict
    except Exception as e:
        logger.error(f"Error in get_3d_target_location: {e}")
        # Return default values if function fails
        return {"depth": 1000.0, "Yaw": 0.0, "Pitch": 0.0, "imgPoints": imgPoints}


@dataclass
class ImageRect:
    points: np.ndarray

    @property
    def center(self):
        return np.average(self.points, axis=0)

    @property
    def width_vec(self):
        return np.average(self.points[2:, :], axis=0) - np.average(self.points[:2, :], axis=0)

    @property
    def width(self):
        return np.linalg.norm(self.width_vec)

    @property
    def height_vec(self):
        return np.average(self.points[(0, 3), :], axis=0) - np.average(self.points[(1, 2), :], axis=0)

    @property
    def height(self):
        return np.linalg.norm(self.height_vec)

    @property
    def angle(self):
        return 90.0 - np.rad2deg(np.arctan2(self.height_vec[1], self.height_vec[0]))


# find contours and main screening section
def find_contours(config: CVParams, binary, frame, depth_frame, fps):
    global num
    # Use a more efficient contour retrieval mode for better performance
    try:
        # Use a faster contour finding approach
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        first_data = []  # include all potential light bar's contourArea information dict by dict
        second_data = []
        # all potential target's [depth,yaw,pitch,imgPoints(np.array([[bl], [tl], [tr],[br]]))]
        potential_Targets = []
        
        # Skip if no contours found for efficiency
        if len(contours) == 0:
            return potential_Targets
        
        # Filter contours by area first to reduce processing
        # Use numpy for faster filtering
        areas = np.array([cv2.contourArea(contour) for contour in contours])
        valid_indices = np.where(areas >= 5)[0]
        contours = [contours[i] for i in valid_indices]
        
        # Pre-compute threshold values for efficiency
        bar_aspect_ratio_min = config.bar_aspect_ratio_min
        bar_aspect_ratio_max = config.bar_aspect_ratio_max
        bar_z_angle_max = config.bar_z_angle_max
        
        # Optimize first_data construction
        for contour in contours:
            try:
                rect = cv2.minAreaRect(contour)
                # coordinates of the four vertices of the rectangle
                coor = cv2.boxPoints(rect).astype(np.int32)
    
                rect_param = findVerticesOrder(coor)  # output order: [bl,tl,tr,br]
                rect = ImageRect(rect_param)
                
                # Debug visualization - only if showing the stream
                if debug:
                    cv2.circle(frame, rect.points[0], 9, (255, 255, 255), -1)  # test armor_tr
                    cv2.circle(frame, rect.points[1], 9, (0, 255, 0), -1)  # test armor_tl
                    cv2.circle(frame, rect.points[2], 9, (255, 255, 0), -1)  # test bottom left
                    cv2.circle(frame, rect.points[3], 9, (0, 100, 250), -1)  # test bottom left
    
                # Filter by aspect ratio and angle
                aspect_ratio = rect.height / rect.width
                angle = rect.angle
                
                # Combine the comparisons to reduce branch mispredictions
                if (aspect_ratio >= bar_aspect_ratio_min and 
                    aspect_ratio <= bar_aspect_ratio_max and 
                    abs(angle) <= bar_z_angle_max):
                    
                    first_data.append(rect)
                    
                    # Debug visualization - only if showing the stream
                    if debug:
                        box = np.int0(coor)
                        cv2.drawContours(frame, [box], -1, (255, 0, 0), 3)
            except Exception as e:
                logger.error(f"Error processing contour: {e}")
                continue
        
        # Pre-compute values for second-level filtering
        relative_y_delta_max = config.relative_y_delta_max
        relative_height_diff_max = config.relative_height_diff_max
        relative_x_delta_max = config.relative_x_delta_max
        z_delta_max = config.z_delta_max
        
        # Optimize second_data construction
        len_first_data = len(first_data)
        for i in range(len_first_data):
            c = first_data[i]
            c_center = c.center
            c_height = c.height
            c_angle = c.angle
            
            for j in range(i + 1, len_first_data):
                n = first_data[j]
                try:
                    # Calculate all deltas at once
                    n_center = n.center
                    n_height = n.height
                    
                    # Calculate deltas efficiently
                    y_delta = abs(c_center[1] - n_center[1])
                    x_delta = abs(c_center[0] - n_center[0])
                    height_ratio = abs(c_height - n_height) / max(c_height, n_height)
                    angle_delta = abs(c_angle - n.angle)
                    height_avg = (c_height + n_height) / 2
                    
                    # Use combined comparison to reduce branching
                    if (y_delta <= relative_y_delta_max * height_avg and
                        height_ratio <= relative_height_diff_max and
                        x_delta <= relative_x_delta_max * height_avg and
                        angle_delta < z_delta_max):
                        
                        second_data.append((c, n))
                except Exception as e:
                    logger.error(f"Error processing rectangle pair: {e}")
                    continue
        
        # Early return if no pairs found
        if not second_data:
            return potential_Targets
            
        # Optimize target creation - only allocate memory once for the result
        potential_Targets = []
        
        # Process pairs to find targets
        for r1, r2 in second_data:
            try:
                # Calculate only once
                x_diff = abs(r1.points[0][0] - r2.points[2][0])
                y_diff = abs(r1.points[0][1] - r2.points[2][1])
                
                if y_diff <= 3 * x_diff:
                    # Determine left and right bars once
                    left_bar, right_bar = (r1, r2) if r1.points[3][0] <= r2.points[3][0] else (r2, r1)
                    
                    # Calculate vectors once
                    left_side_vec = (left_bar.points[0] - left_bar.points[1]) / 2
                    right_side_vec = (left_bar.points[3] - left_bar.points[2]) / 2
                    
                    # Construct array of points
                    imgPoints = np.array(
                        [left_bar.points[0] + left_side_vec, 
                         left_bar.points[1] - left_side_vec,
                         right_bar.points[2] - right_side_vec, 
                         right_bar.points[3] + right_side_vec],
                        dtype=np.float64)
                    
                    # Get target location
                    target_Dict = get_3d_target_location(
                        imgPoints, frame, depth_frame)
                    
                    # Create target object and add to list
                    target = Target(target_Dict)
                    potential_Targets.append(target)
                    
                    # Debug visualization - only if in debug mode
                    if debug:
                        num += 1
                        cv2.putText(frame, "Potentials:", (int(imgPoints[2][0]), int(imgPoints[2][1]) - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255])
                        center = np.average(imgPoints, axis=0).astype(np.int32)
                        cv2.circle(frame, center, 2, (0, 0, 255), -1)
            except Exception as e:
                logger.error(f"Error creating target: {e}")
                continue
                
        return potential_Targets
        
    except Exception as e:
        logger.error(f"Error in find_contours: {e}")
        return []


def targetsFilter(potential_Targetsets, frame, last_target_x):
    
    # if only one target, return it directly
    if len(potential_Targetsets) == 1:
        return potential_Targetsets[0] # the only target class object
    
    '''
    target with Number & greatest credits wins in filter process
    Credit Consideration: Area, Depth, Pitch, Yaw
    Credit Scale: 1 - 3
    '''
    # Pre-allocate arrays for faster operations
    max_Credit = 0
    best_Target = None
    
    # if the target from last frame exists, filter out the closest one to keep tracking on the same target
    if last_target_x is not None:
        # Calculate all distances at once for performance
        all_distance_diff = []
        for target in potential_Targetsets:
            imgPoints = target.imgPoints
            # current target's x-axis in a 1280*720 frame
            curr_target_x = imgPoints[0][0] + (imgPoints[2][0] - imgPoints[0][0]) / 2
            all_distance_diff.append(abs(curr_target_x - last_target_x))
            
        # Find the index of the minimum distance
        closest_idx = np.argmin(all_distance_diff)
        return potential_Targetsets[closest_idx]

    # if the target from last frame doesn't exist, filter out the best one based on credits
    for target in potential_Targetsets:
        depth = float(target.depth)
        Yaw = float(target.yaw)
        Pitch = float(target.pitch)

        # target with greatest credits wins in filter process;total_Credit = depth credit + angle credit
        depth_Credit = 0
        angle_Credit = 0

        """Assess Depth - use faster conditional logic"""
        if depth < 1800:
            depth_Credit = 5
        elif depth < 2500:
            depth_Credit = 3

        """Assess Angle - use faster conditional logic"""
        if abs(Yaw) < 5 or abs(Pitch) < 10:
            angle_Credit = 100
        elif abs(Yaw) < 10 or abs(Pitch) < 15:
            angle_Credit = 3
        elif abs(Yaw) < 20 or abs(Pitch) < 20:
            angle_Credit = 2
        elif abs(Yaw) < 30 or abs(Pitch) < 30:
            angle_Credit = 1

        """evaluate score"""
        current_credit = depth_Credit + angle_Credit
        if current_credit > max_Credit:
            max_Credit = current_credit
            best_Target = target

    return best_Target


def clipRect(rect_xywh, size):
    x, y, w, h = rect_xywh
    clipped_x, clipped_y = min(max(x, 0), size[0]), min(max(y, 0), size[1])
    return clipped_x, clipped_y, min(max(w, 0), size[0] - clipped_x), min(max(h, 0), size[1] - clipped_y)


def findVerticesOrder(pts):
    ''' sort rectangle points by clockwise '''
    # sort y-axis only - use numpy's direct sorting
    y_sorted_indices = np.argsort(pts[:, 1])
    sort_x = pts[y_sorted_indices, :]
    
    # get top 2 [x,y] and bottom 2 [x,y]
    Bottom = sort_x[2:, :]  # bot
    Top = sort_x[:2, :]  # top

    # Bottom sort: Bottom[0] = bl ;  Bottom[1] = br
    Bottom = Bottom[np.argsort(Bottom[:, 0]), :]

    # Top sort: Top[0] = tl ; Top[1] = tr
    Top = Top[np.argsort(Top[:, 0]), :]

    # Directly stack into a new array to avoid Python loop
    return np.stack([Bottom[0], Top[0], Top[1], Bottom[1]], axis=0)


def float_to_hex(f):
    ''' turn float to hex'''
    return ''.join([f'{byte:02x}' for byte in struct.pack('>f', f)])

def decimalToHexSerial(Yaw, Pitch):
    # Yaw and Pitch to IEEE 754 standard four-byte floating point representation and convert to hexadecimal string
    hex_Yaw = float_to_hex(Yaw)
    hex_Pitch = float_to_hex(Pitch)

    # calculate checksum
    bytes_for_checksum = struct.pack('>ff', Yaw, Pitch) # only checked Yaw & Pitch data so far
    checksum = sum(bytes_for_checksum) % 256
    hex_checksum = f'{checksum:02x}'

    # build hexadecimal data list
    return hex_Yaw, hex_Pitch, hex_checksum
     
def draw_crosshair(frame):
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    color = (0, 255, 0)  # Green color
    thickness = 2
    size = 20
    cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), color, thickness)
    cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
    return frame

def main(camera: CameraSource, target_color: TargetColor, show_stream: str):
    """
    Important commit updates: umature pred-imu; 50 deg limit; HSV red adj; get_imu; MVS arch rebuild ---- Shiao
    """
    try:
        cv_config = CVParams(target_color)
    
        # Create a window for CV parameters if debug mode is active
        if debug:
            cv2.namedWindow("CV Parameters")
            createTrackbarsForParams("CV Parameters", cv_config)
            cv2.resizeWindow("CV Parameters", 800, 180)
    
        '''Initialize variables for tracking and prediction'''
    
        fps = 0
        target_coor = []
        lock = False                    # Flag to indicate if the best target is found
        track_init_frame = None
        last_target_x = None
        last_target_y = None
        # success = False
        tracker = None
        tracking_frames = 0
        max_tracking_frames = 15        # Maximum number of frames to track
    
        max_history_length = 8          # Maximum number of samples for prediction
        # Time in seconds to predict the target's motion into the future
        prediction_future_time = 0.2
        '''
        Maximum time in seconds between history frames
        Should be long enough for a dropped frame or two,
        but not too long to group unrelated detections
        '''
        max_history_frame_delta = 0.15
        target_angle_history = []
    
        # Try to Open serial port for data transmission to STM32, if not found, continue without it
        try:
            ser = serial.Serial('/dev/ttyUSB0', 115200)
            logger.info("Successfully opened serial port")
        except Exception as e:
            ser = None
            logger.warning(f"Failed to open serial port: {str(e)}")
            print("Serial port not available. Running without serial communication.")
    
    
        detect_success = False
        track_success = False
        
        # FPS calculation variables
        fps_counter = 0
        fps_sum = 0
        last_fps_print_time = time.time()
        last_fps = 0
        
        # Cache active_cam_config values for faster access
        cam_offset = np.array(camera.active_cam_config['camera_offset'])
        cam_fx = active_cam_config['fx'] 
        cam_fy = active_cam_config['fy']
        cam_cx = active_cam_config['cx']
        cam_cy = active_cam_config['cy']
        
        # Pre-allocate arrays for visualization to avoid repeated memory allocation
        vis_frame = None
        
        while True:
            try:
                "to calculate fps"
                startTime = time.time()
    
                if debug:
                    updateParamsFromTrackbars("CV Parameters", cv_config)
    
                color_image, depth_image = camera.get_frames()
                
                # Skip processing if we didn't get a valid frame
                if color_image is None:
                    logger.warning("No valid frame received, skipping frame")
                    time.sleep(0.01)  # Short sleep to avoid busy-waiting
                    continue
                
                # Only modify color_image with crosshair if we're going to display it
                if show_stream == 'YES' or show_stream == 'yes':
                    frame = color_image.copy()  # Only copy if we need to display
                    frame = draw_crosshair(frame)
                else:
                    frame = color_image  # Just use reference if we're not displaying
                
                """Do detection"""
                binary, frame = read_morphology(color_image, cv_config)
    
                # get the list with all potential targets' info
                potential_Targetsets = find_contours(cv_config, binary, frame, depth_image, fps)
    
                if potential_Targetsets: # if dectection success
                    detect_success = True
    
                    # filter out the best target
                    final_Target = targetsFilter(potential_Targetsets, frame, last_target_x)
    
                    #extract the target's position and angle
                    depth = float(final_Target.depth)
                    Yaw = float(final_Target.yaw)
                    Pitch = float(final_Target.pitch)
                    imgPoints = final_Target.imgPoints
    
                    '''SORT tracking'''
    
                else: # if detection failed
                    """Prepare Tracking"""
    
                    detect_success = False
                    try:
                        if tracker is not None and tracking_frames < max_tracking_frames:
                            tracking_frames += 1
                            # Update tracker
                            track_success, bbox = tracker.update(color_image)
                        else:
                            track_success = False
        
                        """if Tracking success, Solve Angle & Draw bounding box"""
                        if track_success:
                            # Solve angle
                            target_coor_width = abs(
                                int(final_Target.topRight[0]) - int(final_Target.topLeft[0]))
                            target_coor_height = abs(
                                int(final_Target.topLeft[1]) - int(final_Target.bottomLeft[1]))
                            
                            # bbox format:  (init_x,init_y,w,h)
                            bbox = (final_Target.topLeft[0] - target_coor_width * 0.05, final_Target.topLeft[1], target_coor_width * 1.10,
                                    target_coor_height) # to enlarge the bbox to include the whole target, for better tracking by KCF or others
                            
                            bbox = clipRect(bbox, (color_image.shape[1], color_image.shape[0])) # clip the bbox to fit the frame
                            
                            # Calculate all points at once to avoid repeated calculations
                            x, y, w, h = bbox
                            imgPoints = np.array(
                                [[x, y+h], [x, y], [x+w, y],
                                 [x+w, y+h]], dtype=np.float64)
                                 
                            target_Dict = get_3d_target_location(
                                imgPoints, color_image, depth_image)
                            
                            final_Target.depth = target_Dict["depth"]
                            final_Target.yaw = target_Dict["Yaw"]
                            final_Target.pitch = target_Dict["Pitch"]
                            final_Target.imgPoints = target_Dict["imgPoints"]
        
                            '''draw tracking bouding boxes - only if we're showing the stream'''
                            if show_stream == 'YES' or show_stream == 'yes':
                                p1 = (int(bbox[0]), int(bbox[1]))
                                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                    except Exception as e:
                        logger.error(f"Error in tracking: {e}")
                        track_success = False
    
    
                if detect_success or track_success:
                    try:
                        # store the current target's x-axis, used for detection in the next round
                        last_target_x = imgPoints[0][0] + (imgPoints[2][0] - imgPoints[0][0])/2
        
                        '''
                        Do Prediction
                        '''
        
                        if ser is not None:
                            try:
                                imu_yaw, imu_pitch, imu_roll = get_imu(ser)
                                # Don't print this every frame - too much overhead
                                if time.time() - last_fps_print_time >= 1.0:
                                    print(f"imu data receive: {imu_yaw}, {imu_pitch}, {imu_roll}")
                            except Exception as e:
                                logger.error(f"Error reading IMU data: {e}")
                                imu_yaw, imu_pitch, imu_roll = 0, 0, 0  # Safer defaults
                        else:
                            imu_yaw, imu_pitch, imu_roll = 0, 0, 0  # For testing or when serial is unavailable
        
                        # Apply correction factors once
                        imu_yaw_corrected = imu_yaw * -1.2
                        imu_pitch_corrected = imu_pitch * -1.2
                        global_yaw = imu_yaw_corrected + Yaw
                        global_pitch = imu_pitch_corrected + Pitch
        
                        # Calculate cartesian position once
                        cartesian_pos = spherical_to_cartesian(global_yaw, global_pitch, depth) - cam_offset
        
                        # Only do visualization if angles are in range and we're showing the stream
                        if (-30 < Pitch < 30) and (-45 < Yaw < 45):
                            if show_stream == 'YES' or show_stream == 'yes':
                                # Draw visualization for target
                                cv2.line(frame, (int(imgPoints[1][0]), int(imgPoints[1][1])),
                                         (int(imgPoints[3][0]), int(imgPoints[3][1])),
                                         (33, 255, 255), 2)
                                cv2.line(frame, (int(imgPoints[2][0]), int(imgPoints[2][1])),
                                         (int(imgPoints[0][0]), int(imgPoints[0][1])),
                                         (33, 255, 255), 2)
                                cv2.putText(frame, str(depth), (90, 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
                                cv2.putText(frame, str(Yaw), (90, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
                                cv2.putText(frame, str(Pitch), (90, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
        
                            # Update target history
                            current_time = time.time()
                            if len(target_angle_history) < 1 or current_time - target_angle_history[-1][0] > max_history_frame_delta:
                                target_angle_history = [(current_time, *cartesian_pos)]
                            else:
                                target_angle_history.append((current_time, *cartesian_pos))
        
                            if len(target_angle_history) > max_history_length:
                                target_angle_history = target_angle_history[-max_history_length:]
        
                            # Do prediction if we have enough history
                            if len(target_angle_history) >= 2:
                                # Extract history arrays efficiently using numpy
                                target_history_array = np.array(target_angle_history)
                                time_hist_array = target_history_array[:, 0] - target_history_array[0, 0]
                                x_hist_array = target_history_array[:, 1]
                                y_hist_array = target_history_array[:, 2]
                                z_hist_array = target_history_array[:, 3]
        
                                degree = 1  # if len(target_angle_history) == 2 else 2
        
                                # Generate weights linearly
                                weights = np.linspace(float(max_history_length) - len(time_hist_array) + 1.0, 
                                                     float(max_history_length) + 1.0, 
                                                     len(time_hist_array))
                                                     
                                predict_time = time_hist_array[-1] + prediction_future_time
                                
                                # Do predictions
                                predicted_x = poly_predict(time_hist_array, x_hist_array, degree,
                                                           predict_time, weights=weights)
                                predicted_y = poly_predict(time_hist_array, y_hist_array, degree,
                                                           predict_time, weights=weights)
                                predicted_z = poly_predict(time_hist_array, z_hist_array, degree,
                                                           predict_time, weights=weights)
        
                                predicted_yaw, predicted_pitch, _ = cartesian_to_spherical(
                                    np.array([predicted_x, predicted_y, predicted_z]))
                            else:
                                predicted_yaw, predicted_pitch = global_yaw, global_pitch
        
                            # Visualization for prediction - only if showing stream
                            if show_stream == 'YES' or show_stream == 'yes':
                                # Calculate points for visualization
                                current_point_coords = (int(cam_fx * math.tan(math.radians(Yaw)) + cam_cx),
                                                        int(cam_fy * math.tan(math.radians(-Pitch)) + cam_cy))
                                predicted_point_coords = (int(cam_fx * math.tan(math.radians(predicted_yaw - imu_yaw_corrected)) + cam_cx),
                                                          int(cam_fy * math.tan(math.radians(-(predicted_pitch - imu_pitch_corrected))) + cam_cy))
                                cv2.line(frame, current_point_coords,
                                         predicted_point_coords, (255, 255, 255), 2)
        
                            # Calculate relative prediction values
                            relative_pred_yaw = predicted_yaw - imu_yaw_corrected
                            relative_pred_pitch = predicted_pitch - imu_pitch_corrected
        
                            # Clamp prediction values
                            relative_pred_yaw = max(-50, min(50, relative_pred_yaw))  
                            relative_pred_pitch = max(-50, min(50, relative_pred_pitch))
        
                            # Convert to radians for sending
                            Yaw_rad = np.deg2rad(Yaw)
                            Pitch_rad = np.deg2rad(Pitch)
        
                            # Only print this once per second, not every frame
                            if time.time() - last_fps_print_time >= 1.0:
                                print(f"imu data send: {Yaw_rad}, {Pitch_rad}, {detect_success}")
        
                            # Send data to serial port if available
                            if ser is not None:
                                try:
                                    hex_Yaw, hex_Pitch, hex_checksum = decimalToHexSerial(Yaw_rad, Pitch_rad)
                                    send_data(ser, hex_Yaw, hex_Pitch, hex_checksum, detect_success)
                                except Exception as e:
                                    logger.error(f"Error sending data to serial port: {e}")
                        else:
                            logger.warning(f"Angle(s) exceed limits: Pitch: {Pitch}, Yaw: {Yaw}")
                    except Exception as e:
                        logger.error(f"Error in processing detected target: {e}")
    
                else:
                    # Tracking failure - only show message if we're displaying the stream
                    if show_stream == 'YES' or show_stream == 'yes':
                        cv2.putText(frame, "Tracking failure detected", (600, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    
                    # send failure data(send 0 degree to make gimbal stop)
                    if ser is not None:
                        try:
                            hex_Yaw, hex_Pitch, hex_checksum=decimalToHexSerial(0, 0)
                            send_data(ser, hex_Yaw, hex_Pitch, hex_checksum,detect_success)
                        except Exception as e:
                            logger.error(f"Error sending failure data to serial port: {e}")
    
    
                # Only draw UI elements if we're showing the stream
                if show_stream == 'YES' or show_stream == 'yes':
                    cv2.circle(frame, (720, 540), 2, (255, 255, 255), -1)
                    cv2.putText(frame, 'Depth: ', (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
                    cv2.putText(frame, 'Yaw: ', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
                    cv2.putText(frame, 'Pitch: ', (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
                    cv2.putText(frame, 'FPS: ', (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
                    cv2.putText(frame, str(int(last_fps)), (90, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
    
                    cv2.imshow("original", frame)
                    # Use waitKey(1) for maximum speed
                    cv2.waitKey(1)
    
                # Calculate FPS - but only print occasionally to reduce overhead
                endtime = time.time()
                frame_time = endtime - startTime
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Accumulate FPS
                fps_counter += 1
                fps_sum += current_fps
                
                # Print FPS once per second
                if endtime - last_fps_print_time >= 1.0:
                    last_fps = fps_sum / fps_counter if fps_counter > 0 else 0
                    print(f"FPS: {last_fps:.2f}")
                    fps_counter = 0
                    fps_sum = 0
                    last_fps_print_time = endtime
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, exiting")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Continue running even if there's an error
                time.sleep(0.1)  # Short sleep to avoid busy-waiting
                continue
    except Exception as e:
        logger.error(f"Fatal error in main function: {e}")
        # Allow the exception to propagate and terminate the program


if __name__ == "__main__":
    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-color', required=True, type=str, choices=[val.value for val in TargetColor],
                        help='The armor board light color to detect')
    parser.add_argument('--recording-source', type=pathlib.Path,
                        help='Path to input video recordings')
    parser.add_argument('--recording-dest', type=pathlib.Path,
                        help='Path to record camera video to (MP4 format)')
    parser.add_argument('--debug', action='store_true',
                        help='Show intermediate results and debug output')
    parser.add_argument('--show-stream', type=str, choices=['YES', 'NO'], default='NO',
                        help='Display the camera stream (YES or NO)')


    args = parser.parse_args()

    # set up logger
    logger = logging.getLogger(__name__)
    debug: bool = args.debug
    logger.setLevel('DEBUG' if debug else 'INFO')

    args.target_color = TargetColor(args.target_color)
    num = 0  # for collecting dataset, pictures' names

    # choose camera params - use Intel RealSense D435I config based on detected hardware
    try:
        camera = CameraSource(camera_params['Intel RealSense D435I'], args.target_color.value,
                            recording_source=args.recording_source, recording_dest=args.recording_dest)
        
        active_cam_config = camera.active_cam_config
        main(camera, args.target_color, args.show_stream)
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        print("Trying fallback to generic camera...")
        try:
            camera = CameraSource(camera_params['Generic Webcam'], args.target_color.value,
                                recording_source=args.recording_source, recording_dest=args.recording_dest)
            active_cam_config = camera.active_cam_config
            main(camera, args.target_color, args.show_stream)
        except Exception as e:
            print(f"Failed to initialize generic camera: {e}")
            print("No camera device found. Please check your camera connection.")
