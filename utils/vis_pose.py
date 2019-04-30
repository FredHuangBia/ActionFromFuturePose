import cv2
import numpy as np


def single_2D_pose(pose, offset = [0,0,0], scale = [1,1,1]):
    #expects a single (V x C=3) matrix
    transformed_pose = (pose*scale)+offset
    
    #convert pose to 2D
    rvec = np.array([0., 0., 0.])
    tvec = np.array([0., 0., 50.])
    cameraMatrix = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
    distCoeffs = None
    _2D_pose,_ = cv2.projectPoints(transformed_pose, rvec, tvec, cameraMatrix, distCoeffs)
    
    return _2D_pose

def single_pose_image(pose, color=(255,255,255)):
    
    lines = [[0,1], [20,2], [2,3], [20,4], [4,5], [5,6], [6,7], [20,8], [8,9], [9,10], [10,11], [0,12],
             [12,13], [13, 14], [14,15], [0,16], [16,17], [17,18], [18,19], [1,20], [7,21], [7,22], [11,23], [11,24]]
            
    assert pose.shape[1]==3, "Incorrect input shape"
    _2D_pose = (pose[:,:2].reshape(25,2) * 500).astype('int')
    #_2D_pose = (single_2D_pose(pose).reshape(25,2) * 100).astype('int')
    #display on image

    _2D_pose[:,0] -= _2D_pose[:,0].min()
    _2D_pose[:,1] -= _2D_pose[:,1].min()
    
    max_height = _2D_pose[:,0].max() - _2D_pose[:,0].min()
    max_width = _2D_pose[:,1].max() - _2D_pose[:,1].min()
    image_height = int(max_height * 1.5)
    image_width = int(max_width * 1.5)
    shift_height= int(max_height * 0.25)
    shift_width= int(max_width * 0.25)

    '''
    print(image_height)
    print(_2D_pose[:,0].max())
    print(_2D_pose[:,0].min())
    print(image_width)
    print(_2D_pose[:,1].max())
    print(_2D_pose[:,1].min())
    '''
    
    #create image
    image = np.zeros([image_height, image_width])
    
    for line in lines:
        cv2.line(image, (_2D_pose[line[0],1]+shift_width, _2D_pose[line[0],0]+shift_height), (_2D_pose[line[1],1]+shift_width,_2D_pose[line[1],0]+shift_height), color, thickness=5)

    for p in range(_2D_pose.shape[0]):
        cv2.circle(image, (_2D_pose[p,1]+shift_width,_2D_pose[p,0]+shift_height), 2, color, thickness=10)    
        
        
    #transpose image
    image = image.transpose(1,0)
    image = cv2.flip(image, 0)
    kernel = np.ones((5,5),np.float32)/25
    image = cv2.filter2D(image,-1,kernel)
    image = cv2.resize(image,(200,400))
    
    return image

def double_pose_image(pose1, pose2, color1=(255,255,0), color2=(0,255,255)):
    
    lines = [[0,1], [20,2], [2,3], [20,4], [4,5], [5,6], [6,7], [20,8], [8,9], [9,10], [10,11], [0,12],
             [12,13], [13, 14], [14,15], [0,16], [16,17], [17,18], [18,19], [1,20], [7,21], [7,22], [11,23], [11,24]]
            
    _2D_pose1 = (pose1[:,:2].reshape(25,2) * 500).astype('int')
    _2D_pose2 = (pose2[:,:2].reshape(25,2) * 500).astype('int')
    #_2D_pose = (single_2D_pose(pose).reshape(25,2) * 100).astype('int')
    #display on image

    _2D_pose1[:,0] -= _2D_pose1[:,0].min()
    _2D_pose1[:,1] -= _2D_pose1[:,1].min()
    _2D_pose2[:,0] -= _2D_pose2[:,0].min()
    _2D_pose2[:,1] -= _2D_pose2[:,1].min()
    
    image_height = int(max((_2D_pose1[:,0].max() - _2D_pose1[:,0].min()), (_2D_pose2[:,0].max() - _2D_pose2[:,0].min())) * 1.5)
    image_width = int(max((_2D_pose1[:,1].max() - _2D_pose1[:,1].min()), (_2D_pose2[:,1].max() - _2D_pose2[:,1].min())) * 1.5)

    '''
    print(image_height)
    print(_2D_pose[:,0].max())
    print(_2D_pose[:,0].min())
    print(image_width)
    print(_2D_pose[:,1].max())
    print(_2D_pose[:,1].min())
    '''
    
    #create image
    image = np.zeros([image_height, image_width])
    
    for line in lines:
        cv2.line(image, (_2D_pose1[line[0],0], _2D_pose1[line[0],1]), (_2D_pose1[line[1],0],_2D_pose1[line[1],1]), color1, thickness=2)
        cv2.line(image, (_2D_pose2[line[0],0], _2D_pose2[line[0],1]), (_2D_pose2[line[1],0],_2D_pose2[line[1],1]), color2, thickness=2)

    for p in range(_2D_pose1.shape[0]):
        cv2.circle(image, (_2D_pose1[p,0],_2D_pose1[p,1]), 2, color1, thickness=2)    

    for p in range(_2D_pose2.shape[0]):
        cv2.circle(image, (_2D_pose2[p,0],_2D_pose2[p,1]), 2, color2, thickness=2)    
        
    return image

    
def main():
        
    sample_pose = np.array([[[ -9.6925],[ -9.7138],  [ -9.7342],  [ -9.6590],  [ -9.8016],  [ -9.7599],  [ -9.6415],  [ -9.6166],  [ -9.6471],  [ -9.7464],  [ -9.6637],  [ -9.6600],  [ -9.7298],  [ -9.7179],  [ -9.7708],  [ -9.6795],  [ -9.6637],  [ -9.6854],  [ -9.8038],  [ -9.7618],  [ -9.7288],  [ -9.5601],  [ -9.6628],  [ -9.6414],  [ -9.6814]], [[ -9.4176],  [ -9.1423],  [ -8.8762],  [ -8.7473],  [ -9.0178],  [ -9.2648],  [ -9.4551],  [ -9.5253],  [ -8.9135],  [ -8.9440],  [ -9.1617],  [ -9.2640],  [ -9.4258],  [ -9.7777],  [-10.0827],  [-10.1436],  [ -9.4189],  [ -9.8132],  [-10.0374],  [-10.1059],  [ -8.9412],  [ -9.5811],  [ -9.5185],  [ -9.2959],  [ -9.2589]], [[-78.1498],  [-78.2781],  [-78.4164],  [-78.4354],  [-78.4724],  [-78.4383],  [-78.3339],  [-78.2619],  [-78.2031],  [-78.5201],  [-78.3114],  [-78.3320],  [-78.2227],  [-78.0755],  [-77.8429],  [-77.8968],  [-78.1462],  [-78.0775],  [-77.8681],  [-77.9534],  [-78.3797],  [-78.2327],  [-78.2500],  [-78.2995],  [-78.3750]]])

    sample_pose = sample_pose.reshape(3,25).transpose(1,0)
    image = single_pose_image(sample_pose)
    
if __name__ == "__main__":
    main()    