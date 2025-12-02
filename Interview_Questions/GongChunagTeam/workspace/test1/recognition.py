'''
圆环识别与激光打点识别
    题目内容：请识别工创赛场地的三个不同色的同心多层圆环与识别激光点
    题目要求：
    1. 以摄像头画面的左下角为原点，以一个像素为单位，建立坐标系，给出激光点的坐标，
    位于哪个圆环区域，就用哪种颜色输出激光点的坐标，同时使用红点标出激光点中心位置。
    2. 同心圆环，从外到里的分数为：40，60，100（仅考虑实线的圆），请给出激光点所在区域的分数。
'''
'''
░░░░░░░▐█▀█▄░░░░░░░░░░▄█▀█▌
░░░░░░░█▐▓░█▄░░░░░░░▄█▀▄▓▐█
░░░░░░░█▐▓▓░████▄▄▄█▀▄▓▓▓▌█
░░░░░▄█▌▀▄▓▓▄▄▄▄▀▀▀▄▓▓▓▓▓▌█
░░░▄█▀▀▄▓█▓▓▓▓▓▓▓▓▓▓▓▓▀░▓▌█
░░█▀▄▓▓▓███▓▓▓███▓▓▓▄░░▄▓▐█▌
░█▌▓▓▓▀▀▓▓▓▓███▓▓▓▓▓▓▓▄▀▓▓▐█
▐█▐██▐░▄▓▓▓▓▓▀▄░▀▓▓▓▓▓▓▓▓▓▌█▌
█▌███▓▓▓▓▓▓▓▓▐░░▄▓▓███▓▓▓▄▀▐█
█▐█▓▀░░▀▓▓▓▓▓▓▓▓▓██████▓▓▓▓▐█▌
▓▄▌▀░▀░▐▀█▄▓▓██████████▓▓▓▌█
'''
# 导入
import cv2
import numpy as np

# 上视频及判别
vedio_input = cv2.VideoCapture('input.avi')
if not vedio_input.isOpened():
    print('打开视频错误')
    exit()

_,frame_1 = vedio_input.read()
if _:
    frame_width = frame_1.shape[1]
    frame_height = frame_1.shape[0]
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writers = cv2.VideoWriter('output.avi',fourcc,30,frame_size)


def Circleheart(frame_color,x,y):
    frame_color = 255 - frame_color
    R_circleL, galgame = cv2.findContours(frame_color,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if R_circleL :
        valid_contours = []
        min_area = 100
        for cnt in R_circleL:
            if cv2.contourArea(cnt) >= min_area:
                valid_contours.append(cnt)

        max_contour = max(valid_contours, key=cv2.contourArea)
        (final_cx, final_cy), _ = cv2.minEnclosingCircle(max_contour)
        
        final_center = [int(final_cx), int(final_cy)]
        # print(f"final_center{final_center}")
        return ychange(final_center,x,y)

# 还的有个坐标变换
def ychange(final_center,x,y):
    using_center = [0,0]
    using_center[0] = x+final_center[0]
    using_center[1] = y+final_center[1]
    final_center[0] = x+final_center[0]
    final_center[1] = 479-y-final_center[1]
    return using_center,final_center

def DistanceDetect(center,dotpoint,bgr):
    dotpoint = np.array(dotpoint)
    centerpoint = np.array(center)
    distance = int(np.linalg.norm(centerpoint - dotpoint))
    ifdraw = True
    color = (0,0,0)
    result = 0
    
    if bgr == 1:
        color = (0,0,255)
    elif bgr == 2:
        color = (0,255,0) 
    elif bgr == 3:
        color = (255,0,0)
    # print(bgr,distance)
    if distance > 35 and distance <= 45:
        result = 40
    elif distance > 27 and distance <= 35:
        result = 60
    elif distance <= 27:
        result = 100
    else:
        ifdraw = False
        color  = (0,0,0)
    return ifdraw,result,color

# 基础数据
r_bound_x = 10
r_bound_y = 155
r_boundsize_x = 140
r_boundsize_y = 180

g_bound_x = 175
g_bound_y = 170
g_boundsize_x = 140
g_boundsize_y = 180

b_bound_x = 340
b_bound_y = 160
b_boundsize_x = 140
b_boundsize_y = 180
# 帧处理
while True:
    rat,frame_1 = vedio_input.read()
    if not rat:
        break
    ifdraw = True
    # --------------------------------------------处理---------------------------------------------
    # 灰度
    frame_2 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    
    # ROI区域
    frame_red = frame_2[r_bound_y:r_bound_y+r_boundsize_y,r_bound_x:r_bound_x+r_boundsize_x]
    frame_green = frame_2[g_bound_y:g_bound_y+g_boundsize_y,g_bound_x:g_bound_x+g_boundsize_x]
    frame_blue = frame_2[b_bound_y:b_bound_y+b_boundsize_y,b_bound_x:b_bound_x+b_boundsize_x]
    
    # 阈值处理
    frame_red_2 = cv2.adaptiveThreshold(frame_red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=11)
    frame_green_2 = cv2.adaptiveThreshold(frame_green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=11)
    frame_blue_2 = cv2.adaptiveThreshold(frame_blue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=11)
    frame_3_ret,frame_3 = cv2.threshold(frame_2, 240, 255, cv2.THRESH_BINARY)
    # 圆心检测
    r_center,biao_r_center = Circleheart(frame_red_2,r_bound_x,r_bound_y)
    # print("r",r_center)
    g_center,biao_g_center = Circleheart(frame_green_2,g_bound_x,g_bound_y)
    # print("g",g_center)
    b_center,biao_b_center = Circleheart(frame_blue_2,b_bound_x,b_bound_y)
    # print("b",b_center)
    # 点的坐标获取
    Point_contours, hahahahaha = cv2.findContours(frame_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if Point_contours != ():
        Point = cv2.moments(Point_contours[0])
        if Point["m00"] != 0:
            Point_x = int(Point["m10"] / Point["m00"])
            Point_y = int(Point["m01"] / Point["m00"])
            # print(f"({Point_x},{450-Point_y})")
            GPoint = np.array([Point_x,Point_y])
            
            # 绘制
            
            cv2.circle(frame_1, r_center, 4, (0,0,255), -1)
            cv2.circle(frame_1, r_center, 45, (0,0,255), 1, cv2.LINE_AA)
            cv2.circle(frame_1, r_center, 35, (0,0,255), 1, cv2.LINE_AA)
            cv2.circle(frame_1, r_center, 27, (0,0,255), 1, cv2.LINE_AA)
            
            cv2.circle(frame_1, g_center, 4, (0,255,0), -1)
            cv2.circle(frame_1, g_center, 45, (0,255,0), 1, cv2.LINE_AA)
            cv2.circle(frame_1, g_center, 35, (0,255,0), 1, cv2.LINE_AA)
            cv2.circle(frame_1, g_center, 27, (0,255,0), 1, cv2.LINE_AA)
            
            cv2.circle(frame_1, b_center, 4, (255,0,0), -1)
            cv2.circle(frame_1, b_center, 45, (255,0,0), 1, cv2.LINE_AA)
            cv2.circle(frame_1, b_center, 35, (255,0,0), 1, cv2.LINE_AA)
            cv2.circle(frame_1, b_center, 27, (255,0,0), 1, cv2.LINE_AA)
            
            # cv2.circle(frame_1, GPoint, 4, (0,0,0), -1)
            # 所在区域判断
            # print(Point_x)
            if Point_x >=0 and Point_x <150:
                ifdraw,result,color = DistanceDetect(r_center,GPoint,1)
                if ifdraw == True:
                    cv2.circle(frame_1, GPoint, 4, color, -1)
            elif Point_x >150 and Point_x <=330:
                ifdraw,result,color = DistanceDetect(g_center,GPoint,2)
                if ifdraw == True:
                    cv2.circle(frame_1, GPoint, 4, color, -1)
            elif Point_x >330:
                ifdraw,result,color = DistanceDetect(b_center,GPoint,3)
                if ifdraw == True:
                    cv2.circle(frame_1, GPoint, 4, color, -1)

            cv2.putText(frame_1, f"Position:{GPoint}------score:{result}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 1, cv2.LINE_AA)
            
    else:
        cv2.circle(frame_1, r_center, 4, (0,0,255), -1)
        cv2.circle(frame_1, r_center, 45, (0,0,255), 1, cv2.LINE_AA)
        cv2.circle(frame_1, r_center, 35, (0,0,255), 1, cv2.LINE_AA)
        cv2.circle(frame_1, r_center, 27, (0,0,255), 1, cv2.LINE_AA)
        
        cv2.circle(frame_1, g_center, 4, (0,255,0), -1)
        cv2.circle(frame_1, g_center, 45, (0,255,0), 1, cv2.LINE_AA)
        cv2.circle(frame_1, g_center, 35, (0,255,0), 1, cv2.LINE_AA)
        cv2.circle(frame_1, g_center, 27, (0,255,0), 1, cv2.LINE_AA)
        
        cv2.circle(frame_1, b_center, 4, (255,0,0), -1)
        cv2.circle(frame_1, b_center, 45, (255,0,0), 1, cv2.LINE_AA)
        cv2.circle(frame_1, b_center, 35, (255,0,0), 1, cv2.LINE_AA)
        cv2.circle(frame_1, b_center, 27, (255,0,0), 1, cv2.LINE_AA)
        
    # --------------------------------------------处理---------------------------------------------
    cv2.imshow('test1',frame_1)
    # 保存
    writers.write(frame_1)

    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放
vedio_input.release()
cv2.destroyAllWindows()