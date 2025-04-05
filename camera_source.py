import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2
from camera_params import camera_params, DepthSource
from hik_driver import *

logger = logging.getLogger(__name__)

RS_DEPTH_CAPTURE_RES = (640, 480)

# Unified image acquisition class for different types of cameras
class CameraSource:
    def __init__(self, default_config: Dict, target_color: str, cv_device_index: int = 0,
                 recording_source: Optional[Path] = None, recording_dest: Optional[Path] = None):
        assert recording_source is None or recording_dest is None
        self.recording_source=recording_source
        self._rs_pipeline = None
        self._rs_frame_aligner = None
        self._cv_color_cap = None
        self._cv_depth_cap = None
        self.hik_frame_cap = None
        self.color_frame_writer = None
        self.depth_frame_writer = None
        self.active_cam_config = None
        self.hik_frame_init=None

        if recording_source is None:
            self.active_cam_config = default_config

            # First try to use Intel RealSense camera
            try:
                import pyrealsense2 as rs
                # Configure depth and color streams
                pipeline = rs.pipeline()
                config = rs.config()

                # Get device product line for setting a supporting resolution
                pipeline_wrapper = rs.pipeline_wrapper(pipeline)
                pipeline_profile = config.resolve(pipeline_wrapper)
                device = pipeline_profile.get_device()
                device_name = str(device.get_info(rs.camera_info.name))

                logger.info(f"Detected RealSense camera: {device_name}")
                
                if device_name in camera_params:
                    self.active_cam_config = camera_params[device_name]
                    logger.info(f"Using configuration for {device_name}")
                else:
                    # If device name not found, check for D435I specifically
                    if "D435I" in device_name or "435i" in device_name:
                        self.active_cam_config = camera_params['Intel RealSense D435I']
                        logger.info(f"Using Intel RealSense D435I configuration")
                    else:
                        logger.warning(
                            f'Unknown device name: "{device_name}". Falling back to default configuration.')

                # Reset configuration to ensure clean start
                config = rs.config()
                
                # Configure color stream - use a more compatible resolution/format
                config.enable_stream(
                    rs.stream.color, 
                    640, 480,  # Use standard resolution that's widely supported
                    rs.format.bgr8, 
                    30  # Use standard frame rate
                )

                if self.active_cam_config['depth_source'] == DepthSource.STEREO:
                    config.enable_stream(
                        rs.stream.depth, 
                        640, 480,  # Use standard resolution 
                        rs.format.z16, 
                        30  # Standard frame rate
                    )
                    frame_aligner = rs.align(rs.stream.color)
                else:
                    frame_aligner = None

                # Start streaming
                try:
                    pipeline_profile = pipeline.start(config)
                    logger.info("Successfully started RealSense pipeline")
                    
                    # Get the sensors
                    depth_sensor = None
                    color_sensor = None  # Define here to avoid "used before assignment" error
                    
                    for sensor in pipeline_profile.get_device().query_sensors():
                        if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                            color_sensor = sensor
                        elif sensor.get_info(rs.camera_info.name) == 'Stereo Module':
                            depth_sensor = sensor
    
                    # Set the exposure for the color sensor
                    if depth_sensor is not None:
                        # Enable auto exposure for depth sensor
                        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
                    
                    # Set color sensor options
                    if color_sensor is not None:
                        try:
                            color_sensor.set_option(
                                rs.option.exposure, self.active_cam_config['exposure'][target_color])
                            logger.info(f"Set exposure to {self.active_cam_config['exposure'][target_color]}")
                        except Exception as e:
                            logger.warning(f"Failed to set exposure: {e}")
                    
                    self._rs_pipeline = pipeline
                    self._rs_frame_aligner = frame_aligner
                    
                    logger.info(f"Successfully initialized Intel RealSense camera: {device_name}")
                except RuntimeError as ex:
                    logger.error(f"Failed to start pipeline: {ex}")
                    raise
                
            except ImportError:
                logger.warning(
                    'Intel RealSense backend is not available; pyrealsense2 could not be imported')
            except RuntimeError as ex:
                logger.error(f"RealSense error: {ex}")
                if len(ex.args) >= 1 and 'No device connected' in ex.args[0]:
                    logger.warning('No RealSense device was found')
                else:
                    logger.error(f"Failed to initialize RealSense camera: {ex}")
                    
            # If RealSense is not available, try using OpenCV camera
            if self._rs_pipeline is None:
                cap = cv2.VideoCapture()

                # Try to open standard camera
                if cap.open(cv_device_index):
                    logger.info("Using OpenCV camera")
                    cap.set(cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    cap.set(cv2.CAP_PROP_EXPOSURE,
                            self.active_cam_config['exposure'][target_color])
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                            self.active_cam_config['capture_res'][0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                            self.active_cam_config['capture_res'][1])
                    cap.set(cv2.CAP_PROP_FPS, self.active_cam_config['frame_rate'])
                    self._cv_color_cap = cap
                # If built-in camera not available, try HikRobot as last resort
                else:
                    try:
                        logger.info("Trying to use HikRobot camera as fallback")
                        self.hik_frame_init = hik_init()
                        logger.info("Using HikRobot camera")
                    except Exception as e:
                        logger.error(f"Failed to initialize HikRobot camera: {e}")
                        logger.error("No camera device found")

        else:
            cam_config_path = recording_source.with_name(
                recording_source.name + '.config.json')
            with open(cam_config_path, 'r', encoding='utf8') as cam_config_file:
                self.active_cam_config = json.load(cam_config_file)

            # color_frame_path = recording_source.with_name(
            #     recording_source.name + '.color.mp4')
            # depth_frame_path = recording_source.with_name(
            #     recording_source.name + '.depth.mp4')
            self._cv_color_cap = cv2.VideoCapture(str(recording_source))
            # self._cv_depth_cap = cv2.VideoCapture(str(depth_frame_path))

        self.color_frame_writer = self.depth_frame_writer = None
        if recording_dest is not None:
            cam_config_path = recording_dest.with_name(
                recording_dest.name + '.config.json')
            with open(cam_config_path, 'w', encoding='utf8') as cam_config_file:
                json.dump(self.active_cam_config, cam_config_file)

            # Note that using this codec requires video files to have a .mp4 extension, otherwise
            # writing frames will fail silently.
            codec = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
            color_frame_path = recording_dest.with_name(
                recording_dest.name + '.color.mp4')
            self.color_frame_writer = cv2.VideoWriter(str(color_frame_path), codec,
                                                      self.active_cam_config['frame_rate'],
                                                      self.active_cam_config['capture_res'])
            if self._rs_pipeline is not None:
                depth_frame_path = recording_dest.with_name(
                    recording_dest.name + '.depth.mp4')
                self.depth_frame_writer = cv2.VideoWriter(str(depth_frame_path), codec,
                                                          self.active_cam_config['frame_rate'],
                                                          self.active_cam_config['capture_res'])

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.recording_source is not None:
            # Read from recording source
            ret, color_image = self._cv_color_cap.read()
            if not ret:
                color_image = None

            if self._cv_depth_cap is not None:
                ret, depth_image = self._cv_depth_cap.read()
                if not ret:
                    depth_image = None
            else:
                depth_image = None

        elif self._rs_pipeline is not None:
            frames = self._rs_pipeline.wait_for_frames()

            if self._rs_frame_aligner is not None:
                frames = self._rs_frame_aligner.process(frames)

            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
            else:
                color_image = None

            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
            else:
                depth_image = None

        elif self.hik_frame_init is not None:
            color_image = read_hik_frame(self.hik_frame_init)
            depth_image = None

        elif self._cv_color_cap is not None:
            ret, color_image = self._cv_color_cap.read()
            if not ret:
                color_image = None

            if self._cv_depth_cap is None:
                depth_image = None
            else:
                ret, depth_image = self._cv_depth_cap.read()
                if ret:
                    B, G, R = cv2.split(depth_image)
                    depth_image = (B.astype(np.uint16) << 8) + \
                                  (G.astype(np.uint16) << 12) + R.astype(np.uint16)
                else:
                    depth_image = None


        else:
            raise RuntimeError('No image source available')

        # test = cv2.split(cv2.resize(color_image, RS_DEPTH_CAPTURE_RES))
        # depth_image = (test[0].astype(np.uint16) << 8) + test[2].astype(np.uint16)

        if self.color_frame_writer is not None and color_image is not None:
            self.color_frame_writer.write(color_image)

        if self.depth_frame_writer is not None and depth_image is not None:
            storage_format_image = cv2.merge(
                [(depth_image >> 8).astype(np.uint8), ((depth_image >> 12) % 16).astype(np.uint8),
                 (depth_image % 16).astype(np.uint8)])
            self.depth_frame_writer.write(storage_format_image)

        return color_image, depth_image

    def __del__(self):
        if self._rs_pipeline is not None:
            self._rs_pipeline.stop()

        if self.color_frame_writer is not None:
            self.color_frame_writer.release()

        if self.depth_frame_writer is not None:
            self.depth_frame_writer.release()

        if self.hik_frame_init is not None:
            hik_close(self.hik_frame_init)
