player_detection.py — main entry point. Run this to process a video.
  - Loads player detector + CLIP + ball detector + keypoint analytics                                                             
  - Tracks players with ByteTrack, colors boxes by team jersey                                                                                                        
  - Writes two output videos: player_tracking.mp4 and analytics_tracking.mp4

  - Dataset: https://universe.roboflow.com/neel-ckwq4/cricket-players-mugu4
    Dataset size: ~ 2.6k 
  - training - 1823
  - Base model of yolov8 identifies person including crowd, we fine tuned model to identify only players on field                                                                                        
                                                                                                                                                                      
  team_player_assignment.py — jersey color classifier (called by player_detection)                                                                                    
  - Asks you to enter team colors at startup                                                                                                                          
  - Uses CLIP to decide which team each player belongs to                                                                                                             
  - Returns -1 (no box drawn) if player doesn't match any team color                                                                                                  
                                                                                                                                                                      
  ball_detector.py — ball tracking (called by player_detection)                                                                                                       
  - Pass 1: scans whole video, records where ball was detected each frame                                                                                             
  - Fills gaps ≤15 frames with linear interpolation                                                                                                                   
  - Draws red triangle (real) or orange triangle (interpolated) per frame
  - Dataset: https://universe.roboflow.com/cricket-rodct/cricket-oqrza
    Dataset size: ~ 3.4k 
  - training - 2679
                                                                                     
                                                                                                                                                                      
  WIP:
  keypoint_detection.py — analytics layer (called by player_detection)                                                                                           
  - Detects pitch keypoints → computes homography → converts pixels to real-world metres                                                                              
  - Renders bird's-eye tactical view alongside original frame                                                                                                         
  - Exports: player stats, heatmaps, ball speed report, pitch map

  - Dataset used to fine tune - https://universe.roboflow.com/nishith-n6waq/cricket-pitch-keypoints/browse?queryText=split%3Atrain&pageSize=50&startingIndex=0&browseQuery=true                                                                                                  
                                                                                                       
 Test file:                                                                                                                                                               
  probe_keypoints.py — one-time diagnostic                                                                                                                            
  - Runs keypoint_detection.pt on first frame, prints keypoint indices + saves annotated image                                                                      
  - Use this to figure out which index = which pitch location






<b>output - </b>

will be genrated in op folder
