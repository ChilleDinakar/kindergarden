import math

def get_finger_count(lm):
    tips = [4, 8, 12, 16, 20]
    pip  = [3, 6, 10, 14, 18]
    count = 0
    # Thumb
    def dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    if dist(lm[4], lm[17]) > dist(lm[2], lm[17]):
        count += 1
        
    # Other 4 fingers
    for i in range(1, 5):
        if lm[tips[i]].y < lm[pip[i]].y:
            count += 1
    return count
