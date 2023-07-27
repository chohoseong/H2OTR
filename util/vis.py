import cv2

def visualize(cv_img, img_points, mode='left'):
    if mode == 'left':
        color = (255,0,0)
    else:
        color = (0,0,255)
    line_thickness = 2
    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[2][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[3][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[4][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[5][:-1]), tuple(
        img_points[6][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[6][:-1]), tuple(
        img_points[7][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[7][:-1]), tuple(
        img_points[8][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[9][:-1]), tuple(
        img_points[10][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[10][:-1]), tuple(
        img_points[11][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[11][:-1]), tuple(
        img_points[12][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[13][:-1]), tuple(
        img_points[14][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[14][:-1]), tuple(
        img_points[15][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[15][:-1]), tuple(
        img_points[16][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[17][:-1]), tuple(
        img_points[18][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[18][:-1]), tuple(
        img_points[19][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[19][:-1]), tuple(
        img_points[20][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[1][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[5][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[9][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[13][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[17][:-1]), color, line_thickness)

    return cv_img

def visualize_obj(cv_img, img_points):
    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[2][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[3][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[4][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[4][:-1]), tuple(
        img_points[1][:-1]), (0, 255, 0), 5)

    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[5][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[6][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[7][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[4][:-1]), tuple(
        img_points[8][:-1]), (0, 255, 0), 5)

    cv2.line(cv_img, tuple(img_points[5][:-1]), tuple(
        img_points[6][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[6][:-1]), tuple(
        img_points[7][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[7][:-1]), tuple(
        img_points[8][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[8][:-1]), tuple(
        img_points[5][:-1]), (0, 255, 0), 5)

    return cv_img

def put_text_on_image(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_color = (255, 255, 255)
    font_thickness = 2

    text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]
    text_x = img.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10

    img = cv2.putText(img, text, (text_x, text_y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
    
    return img