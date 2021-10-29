import cv2 as cv
import numpy as np
import serial

ser = serial.Serial('/dev/ttyS0', 4800, timeout=0.0001)
ser.open
ser.flush()  # 시리얼 수신 데이터 버리기

def line_tracing():           # 라인트레이싱
    global flag
    global line_flag
    print(flag, "라인 트레이싱")
    print("line_flag: ", line_flag)

    if line_flag == 0:
        ser.write(serial.to_bytes([117]))



def line_detecting():         # 라인 탐색
    global flag
    print(flag, "라인 탐색")


def color_detecting(img, sig, col):        # 색 찾기
    global flag
    global signal
    global color

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    img_h = hsv[:, :, 0]
    img_s = hsv[:, :, 1]
    img_v = hsv[:, :, 2]

    print(flag, "색 찾기")


def direction_recognizing():  # 방위 인식
    global flag
    print(flag, "방위 인식")


def arrow_recognizing():      # 화살표 인식
    global flag
    print(flag, "화살표 인식")


def area_recognizing():       # 바닥 인식
    global flag
    print(flag, "바닥 인식")


def room_name_recognizing():  # 방 이름 인식
    global flag
    print(flag, "방 이름 인식")
    template = cv.imread("data/ABCD2.png")
    template = cv.resize(template, (500, 350))

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    methods = ['cv2.TM_CCOEFF_NORMED']
    lst = [0, 0, 0, 0, 0, 0, 0, 0]

    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        copy = frame.copy()

        gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
        r, binary = cv.threshold(gray, 100, 255, 0)

        kernel1 = np.ones((2, 2), np.uint8)
        erosion = cv.erode(gray, kernel1, iterations=1)
        dilation = cv.dilate(gray, kernel1, iterations=1)

        morph = dilation - erosion

        r, img_binary = cv.threshold(morph, 30, 255, 0)
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST,
                                               cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cv.drawContours(copy, [cnt], 0, (0, 255, 0), 1)  # 경계 그룹 선으로 표현

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w > 100 and h > 100 and w < 300:
                cv.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 255), 2)
                text = copy[y:(y + h), x:(x + w), ]
                text = cv.resize(text, (90, 90))
                cv.imshow("text", text)
                t = text.copy()
                if text is not 0:
                    th, tw = text.shape[:2]
                    for i, method_name in enumerate(methods):
                        img_draw = template.copy()
                        method = eval(method_name)
                        res = cv.matchTemplate(template, t, method)

                        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                        # print(method_name, min_val, max_val, min_loc, max_loc)

                        if method in [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED]:
                            top_left = max_loc  # 최대값 좌표
                            match_val = max_val  # 최대값(좋은 매칭)

                        bottom_right = (top_left[0] + tw, top_left[1] + th)
                        cv.rectangle(img_draw, top_left, bottom_right, (0, 0, 255), 2)

                        cv.putText(img_draw, str(match_val), top_left,
                                    cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv.LINE_AA)
                        cv.imshow(method_name, img_draw)

                        if top_left[0] < 125 and top_left[1] < 175 and match_val > 0.6:
                            lst[0] += 1
                        elif top_left[0] < 250 and top_left[1] < 175 and match_val > 0.6:
                            lst[1] += 1
                        elif top_left[0] < 375 and top_left[1] < 175 and match_val > 0.6:
                            lst[2] += 1
                        elif top_left[0] < 600 and top_left[1] < 175 and match_val > 0.6:
                            lst[3] += 1
                        elif top_left[0] < 125 and top_left[1] > 175 and match_val > 0.6:
                            lst[4] += 1
                        elif top_left[0] < 250 and top_left[1] > 175 and match_val > 0.6:
                            lst[5] += 1
                        elif top_left[0] < 375 and top_left[1] > 175 and match_val > 0.6:
                            lst[6] += 1
                        elif top_left[0] < 600 and top_left[1] > 175 and match_val > 0.6:
                            lst[7] += 1

                        if sum(lst) > 50:
                            count = lst.index(max(lst))
                            if count == 0:
                                print('red A')
                                break
                            elif count == 1:
                                print('red B')
                                break
                            elif count == 2:
                                print('blue A')
                                break
                            elif count == 3:
                                print('blue B')
                                break
                            elif count == 4:
                                print('red C')
                                break
                            elif count == 5:
                                print('red D')
                                break
                            elif count == 6:
                                print('blue C')
                                break
                            elif count == 7:
                                print('blue D')
                                break
                        break
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        if sum(lst) > 50:
            break

    cv.waitKey(0)
    cv.destroyAllWindows()
    # 인식한 방이름  room_name.append()해야함



def room_name_detecting():  # 방 이름 찾기
    global flag
    print(flag, "방 이름 찾기")


def citizen_recognizing(room_col):    # 시민 인식
    global flag
    global room_color
    print(flag, "시민 인식")


def toward_citizen(room_col):    # 시민 쪽으로 가
    global flag
    global room_color
    print(flag, "시민 쪽으로 가")


def citizen_transporting(room_col):   # 시민 이송
    global flag
    global room_color
    print(flag, "시민 이송")


def dangerous_area_print():   # 확진지역 출력
    global flag
    print(flag, "확진지역 출력")
    print("-------------------------")


if __name__ == '__main__':
    # ser = serial.Serial('/dev/ttyS0', 4800, timeout=0.0001)
    # ser.open
    # ser.flush()   # 시리얼 수신 데이터 버리기

    flag = 0
    signal = 0
    color = " "       # 방위, 시민 인식에 사용하는 색깔
    arrow = " "       # 화살표 방향
    line_flag = 0     # 라인트레이싱 송수신에 이용
    dangerous = []    # 확진구역 방 이름
    area = " "        # 미션구역 색깔
    room_color = " "  # 방 이름 색깔
    room_name = " "   # 방 이름
    mission_flag = 0  # 나가는 시점 확인용 ( = 수행한 미션 개수)
    FPS = 90          # PI CAMERA: 320 x 240 = MAX 90

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    cap.set(5, FPS)

    # h_min, h_max = 10, 30
    # s_min, s_max = 130, 190
    # v_min, v_max = 80, 170

    while 1:
        signal = 0
        if ser.inWaiting() > 0:
            signal = ser.read(1)  # 시리얼 데이터 수신
            signal = int.from_bytes(signal, byteorder='big')

        if signal == 110 or signal == 116:
            flag = 3
        elif signal == 131:
            flag = 8
        elif signal == 133:
            flag = 9
        elif signal == 138:
            flag = 2
        elif signal == 139:
            flag = 12
        elif signal == 128:
            flag = 1

        ret, img = cap.read()    # 이미지 받아오기
        if not ret:
            print("ERROR: 이미지 읽어오기 실패")
            break

        if flag == 0:  continue
        elif flag == 1: line_tracing()
        elif flag == 2: line_detecting()
        elif flag == 3: color_detecting(img, signal, color)
        elif flag == 4: direction_recognizing()         # 나영
        elif flag == 5: arrow_recognizing()             # 나영
        elif flag == 6: area_recognizing()
        elif flag == 7: room_name_recognizing()         # 나영
        elif flag == 8: room_name_detecting()
        elif flag == 9: citizen_recognizing(room_color)
        elif flag == 10: toward_citizen(room_color)
        elif flag == 11: citizen_transporting(room_color)
        elif flag == 12: dangerous_area_print()



