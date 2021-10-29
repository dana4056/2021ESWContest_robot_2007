import cv2 as cv
import numpy as np
import serial
import math

ser = serial.Serial('/dev/ttyS0', 4800, timeout=0.0001)
ser.open
ser.flush()  # 시리얼 수신 데이터 버리기

def line_tracing(img):           # 라인트레이싱
    global flag
    global line_flag
    print(flag, "라인 트레이싱")
    print("line_flag: ", line_flag)

    # h_min, h_max = 20, 40
    # s_min, s_max = 70, 190
    # v_min, v_max = 100, 210

    h_min, h_max = 10, 40
    s_min, s_max = 140, 210
    v_min, v_max = 80, 170

    yellow_low = np.array([10, 140, 80])
    yellow_up = np.array([40, 210, 170])

    ################## 블러링 + HSV변환 ##################
    img = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_h = hsv[:, :, 0]
    img_s = hsv[:, :, 1]
    img_v = hsv[:, :, 2]

    ####################    이진화    ####################
    #binary = cv.inRange(hsv, yellow_low, yellow_up)
    binary = np.zeros((240, 320))  # 검
    binary[((img_h >= h_min) & (img_h <= h_max) &
            (img_s >= s_min) & (img_s <= s_max) &
            (img_v >= v_min) & (img_v <= v_max))] = 1  # 흰

    #################    라인 중점 찾기    ################
    line_idx = np.where(binary == 1)
    center_y = int(sum(line_idx[0]) / len(line_idx[0]))
    center_x = int(sum(line_idx[1]) / len(line_idx[1]))
    # print(center_y, center_x)
    cv.line(img, (center_x, center_y), (center_x, center_y), (255, 0, 0), 5)

    #################    팽창 및 침식    ##################
    mask = cv.dilate(binary, None, iterations=5)  # 팽창
    mask = cv.erode(mask, None, iterations=5)  # 침식

    ###################   윤곽선 검출   ###################
    mask = np.uint8(mask) * 255
    edge = cv.Canny(mask, 50, 200)

    ############## 직선 구하기 및 기울기 계산  ##############
    row_deg, row, col_deg, col = draw_and_compute_lines(edge, img)
    # print(row, col)
    row_x_center = int((row[0] + row[2]) / 2)
    col_x_center = int((col[0] + col[2]) / 2)

    gap_center = abs(row_x_center - col_x_center)
    gap_height = abs(int((row[1] + row[3]) / 2) - col[3])

    # ------------------------ 1 -------------------------
    if line_flag == 0:
        print("send signal 117")
        ser.write(serial.to_bytes([117]))
        line_flag = 1
    # ------------------------ 2 -------------------------
    elif col_deg is not None and row_deg is None and abs(col_deg) < 80:
        if col_deg > 0:
            #pass
            print("좌로 틀어짐 send signal 119", col_deg)
            ser.write(serial.to_bytes([119]))
        elif col_deg < 0:
            #pass
            print("우로 틀어짐 send signal 118", col_deg)
            ser.write(serial.to_bytes([118]))
        line_flag = 0
        flag = 0
    # ------------------------ 3 -------------------------
    elif row_deg is not None and gap_center > 80 and gap_height > 70:
        if row_x_center < col[0]:
            print("ㅓ구간 send signal 120         왼쪽")
            ser.write(serial.to_bytes([120]))
        elif row_x_center > col[0]:
            print("ㅏ구간 send signal 121         오른쪽")
            ser.write(serial.to_bytes([121]))
        line_flag = 0
        flag = 0

    # ------------------------ 4 -------------------------
    elif row_deg is not None and gap_center > 80 and gap_height < 3:
        if row_x_center < col[0]:
            print("ㄱ구간 send signal 122         왼쪽")
            ser.write(serial.to_bytes([122]))
        elif row_x_center > col[0]:
            print("「구간 send signal 123         오른쪽")
            ser.write(serial.to_bytes([123]))
        line_flag = 0
        flag = 6

    # ------------------------ 5 -------------------------
    elif gap_center < 8:
        if arrow == "R":
            #pass
            print("T구간 send signal 125         오른쪽")
            ser.write(serial.to_bytes([125]))
        elif arrow == "L":
            #pass
            print("T구간 send signal 124         왼쪽")
            ser.write(serial.to_bytes([124]))
        line_flag = 0
        flag = 0
    # ------------------------ 6 -------------------------
    elif row_deg is None and center_x < 100:
        print("중앙에서 벗어남(왼쪽으로) send signal 127         라인 위치:", center_x)
        ser.write(serial.to_bytes([127]))
    elif row_deg is None and center_x > 260:
        print("중앙에서 벗어남(오른쪽으로) send signal 126       라인 위치:", center_x)
        ser.write(serial.to_bytes([126]))
        line_flag = 0
        flag = 0
    else:
        print("send signal 117")
        ser.write(serial.to_bytes([117]))
        


def line_detecting():         # 라인 탐색
    global flag
    print(flag, "라인 탐색")


def color_detecting(img, col):        # 색 찾기
    global flag
    global signal
    global color
    print(flag, "색 찾기")

    black_low = np.array([0, 0, 0])
    black_up = np.array([0, 0, 0])

    yellow_low = np.array([0, 0, 0])
    yellow_up = np.array([0, 0, 0])

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    if col == 'black':
        # 범위안에 들어가게되면 0으로 만들어주고 나머지는 1
        black_mask = cv.inRange(hsv, black_low, black_up)
        if sum(black_mask) > 50:
            if signal == 116:
                print("arrow find    send signal 111  |",sum(black_mask))
                ser.write(serial.to_bytes([111]))
                flag = 5
            elif signal == 110:
                print("drection sign find    send signal 111  |", sum(black_mask))
                ser.write(serial.to_bytes([111]))
                flag = 4
    if col == 'yellow':
        yellow_mask = cv.inRange(hsv, black_low, black_up)
        if sum(yellow_mask) > 50:
            print("arrow find    send signal ???  |",sum(yellow_mask))
            ser.write(serial.to_bytes(['???']))
            flag = '???'




def direction_recognizing(img):  # 방위 인식
    global flag
    global flag4_lst
    print(flag, "방위 인식")
    template = cv.imread("data/EWSN.jpg")
    template = cv.resize(template, (480, 260))

    methods = ['cv.TM_CCOEFF_NORMED']

    cv.imshow("hi1", img)
    copy = img.copy()

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

                    if top_left[0] < 240 and top_left[1] < 130 and match_val > 0.6:
                        flag4_lst[0] += 1
                    elif top_left[0] > 240 and top_left[1] < 130 and match_val > 0.65:
                        flag4_lst[1] += 1
                    elif top_left[0] < 240 and top_left[1] > 130 and match_val > 0.6:
                        flag4_lst[2] += 1
                    elif top_left[0] > 240 and top_left[1] > 130 and match_val > 0.65:
                        flag4_lst[3] += 1

                    if sum(flag4_lst) > 10:
                        count = flag4_lst.index(max(flag4_lst))
                        if count == 0:
                            print("send signal 112")  # E
                            ser.write(serial.to_bytes([112]))
                            break
                        elif count == 1:
                            print("send signal 113")  # W
                            ser.write(serial.to_bytes([113]))
                            break
                        elif count == 2:
                            print("send signal 114")  # S
                            ser.write(serial.to_bytes([114]))
                            break
                        elif count == 3:
                            print("send signal 115")  # N
                            ser.write(serial.to_bytes([115]))
                            break
    if sum(flag4_lst) > 50:
        print("wait")  # wait
        flag = 0



def arrow_recognizing():      # 화살표 인식
    global flag
    global arrow
    print(flag, "화살표 인식")
    template = cv.imread("data/ARROW.png")
    template = cv.resize(template, (480, 260))

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    methods = ['cv2.TM_CCOEFF_NORMED']
    lst = [0, 0]

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
                text = cv.resize(text, (150, 90))
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

                        if top_left[0] < 240 and match_val > 0.6:
                            lst[0] += 1
                        elif top_left[0] > 240 and match_val > 0.6:
                            lst[1] += 1

                        if sum(lst) > 10:
                            count = lst.index(max(lst))
                            if count == 0:
                                print('left')
                                arrow = "L"
                                break
                            elif count == 1:
                                print('right')
                                arrow = "R"
                                break
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        if sum(lst) > 50:
            print("line_tracing")
            flag = 1
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


def area_recognizing():       # 바닥 인식
    global flag
    global area
    print(flag, "바닥 인식")

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    g_lower = np.array([0, 0, 0])
    g_upper = np.array([0, 0, 0])

    b_lower = np.array([0, 0, 0])
    b_upper = np.array([0, 0, 0])

    lst = [0, 0]

    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        copy = frame.copy()

        if ret:
            convert = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
            g_mask = cv.inRange(convert, g_lower, g_upper)
            b_mask = cv.inRange(convert, b_lower, b_upper)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
            g_mask = cv.dilate(g_mask, kernel, iterations=1)
            g_mask = cv.GaussianBlur(g_mask, (3, 3), 0)
            b_mask = cv.dilate(b_mask, kernel, iterations=1)
            b_mask = cv.GaussianBlur(b_mask, (3, 3), 0)

            g_line, g_ = cv.findContours(g_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            b_line, b_ = cv.findContours(b_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if g_line:
                cv.drawContours(copy, g_line, -1, (0, 255, 0), 3)
                lst[0] += 1
            if b_line:
                cv.drawContours(copy, b_line, -1, (0, 255, 255), 3)
                lst[1] += 1

            if sum(lst) > 100:
                count = lst.index(max(lst))
                if count == 0:
                    print('green_area')
                    area = "G"
                    ser.write(serial.to_bytes([129]))
                    flag = 0
                    break
                elif count == 1:
                    print('black_area')
                    area = "B"
                    ser.write(serial.to_bytes([130]))
                    flag = 0
                    break

            cv.imshow("images", copy)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('영상 끝~~~~')
            break
    cv.waitKey()
    cv.destroyAllWindows()


def room_name_recognizing():  # 방 이름 인식
    global area
    global flag
    global room_name
    global dangerous
    print(flag, "방 이름 찾기")

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
                        elif top_left[0] < 500 and top_left[1] < 175 and match_val > 0.6:
                            lst[3] += 1
                        elif top_left[0] < 125 and top_left[1] > 175 and match_val > 0.6:
                            lst[4] += 1
                        elif top_left[0] < 250 and top_left[1] > 175 and match_val > 0.6:
                            lst[5] += 1
                        elif top_left[0] < 375 and top_left[1] > 175 and match_val > 0.6:
                            lst[6] += 1
                        elif top_left[0] < 500 and top_left[1] > 175 and match_val > 0.6:
                            lst[7] += 1

                        if sum(lst) > 10:
                            count = lst.index(max(lst))
                            if count == 0:
                                print('red A')
                                room_name.append('A')
                                flag = 9
                                if area == "B":
                                    dangerous.append('A')
                                    ser.write(serial.to_bytes([132]))
                                    flag = 0
                                break
                            elif count == 1:
                                print('red B')
                                room_name.append('B')
                                flag = 9
                                if area == "B":
                                    dangerous.append('B')
                                    ser.write(serial.to_bytes([132]))
                                    flag = 0
                                break
                            elif count == 2:
                                print('blue A')
                                room_name.append('A')
                                flag = 9
                                if area == "B":
                                    dangerous.append('A')
                                    ser.write(serial.to_bytes([132]))
                                    flag = 0
                                break
                            elif count == 3:
                                print('blue B')
                                room_name.append('B')
                                flag = 9
                                if area == "B":
                                    dangerous.append('B')
                                    ser.write(serial.to_bytes([132]))
                                    flag = 0
                                break
                            elif count == 4:
                                print('red C')
                                room_name.append('C')
                                flag = 9
                                if area == "B":
                                    dangerous.append('C')
                                    ser.write(serial.to_bytes([132]))
                                    flag = 0
                                break
                            elif count == 5:
                                print('red D')
                                room_name.append('D')
                                flag = 9
                                if area == "B":
                                    dangerous.append('D')
                                    ser.write(serial.to_bytes([132]))
                                    flag = 0
                                break
                            elif count == 6:
                                print('blue C')
                                room_name.append('C')
                                flag = 9
                                if area == "B":
                                    dangerous.append('C')
                                    ser.write(serial.to_bytes([132]))
                                    flag = 0
                                break
                            elif count == 7:
                                print('blue D')
                                room_name.append('D')
                                flag = 9
                                if area == "B":
                                    dangerous.append('D')
                                    ser.write(serial.to_bytes([132]))
                                    flag = 0
                                break
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        if sum(lst) > 50:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


def room_name_detecting():  # 방 이름 찾기
    global flag
    global room_color
    print(flag, "방 이름 찾기")

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    g_lower = np.array([0, 0, 0])
    g_upper = np.array([0, 0, 0])

    b_lower = np.array([0, 0, 0])
    b_upper = np.array([0, 0, 0])

    lst = [0, 0]

    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        copy = frame.copy()

        if ret:
            convert = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
            g_mask = cv.inRange(convert, g_lower, g_upper)
            b_mask = cv.inRange(convert, b_lower, b_upper)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
            g_mask = cv.dilate(g_mask, kernel, iterations=1)
            g_mask = cv.GaussianBlur(g_mask, (3, 3), 0)
            b_mask = cv.dilate(b_mask, kernel, iterations=1)
            b_mask = cv.GaussianBlur(b_mask, (3, 3), 0)

            g_line, g_ = cv.findContours(g_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            b_line, b_ = cv.findContours(b_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if g_line:
                cv.drawContours(copy, g_line, -1, (0, 255, 0), 3)
                lst[0] += 1
            if b_line:
                cv.drawContours(copy, b_line, -1, (0, 255, 255), 3)
                lst[1] += 1

            if sum(lst) > 100:
                count = lst.index(max(lst))
                if count == 0:
                    print('red_room')
                    ser.write(serial.to_bytes([111]))
                    room_color = "R"
                    flag = 7
                    break
                elif count == 1:
                    print('blue_room')
                    ser.write(serial.to_bytes([111]))
                    room_color = "B"
                    flag = 7
                    break
            cv.imshow("images", copy)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('영상 끝~~~~')
            break
    cv.waitKey()
    cv.destroyAllWindows()


def citizen_recognizing(room_col):    # 시민 인식
    ser.write(serial.to_bytes([134]))
    global flag
    global room_color
    print(flag, "시민 인식")

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    b_lower = np.array([100, 30, 30])
    b_upper = np.array([130, 150, 150])

    r_lower = np.array([150, 120, 30])
    r_upper = np.array([190, 220, 150])

    lst = [0, 0]

    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        copy = frame.copy()

        if ret:
            convert = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
            b_mask = cv.inRange(convert, b_lower, b_upper)
            r_mask = cv.inRange(convert, r_lower, r_upper)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
            b_mask = cv.dilate(b_mask, kernel, iterations=1)
            b_mask = cv.GaussianBlur(b_mask, (3, 3), 0)
            r_mask = cv.dilate(r_mask, kernel, iterations=1)
            r_mask = cv.GaussianBlur(r_mask, (3, 3), 0)

            b_line, b_ = cv.findContours(b_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            r_line, r_ = cv.findContours(r_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if b_line:
                cv.drawContours(copy, b_line, -1, (0, 255, 0), 3)
                lst[0] += 1
            if r_line:
                cv.drawContours(copy, r_line, -1, (0, 255, 255), 3)
                lst[1] += 1

            if sum(lst) > 50:
                count = lst.index(max(lst))
                if count == 0:
                    print('blue_citizen')
                    if room_color == "B":
                        ser.write(serial.to_bytes([135]))
                        flag = 10
                    break
                elif count == 1:
                    print('red_citizen')
                    if room_color == "R":
                        ser.write(serial.to_bytes([135]))
                        flag = 10
                    break

            cv.imshow("images", copy)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('영상 끝~~~~')
            break
    cv.waitKey()
    cv.destroyAllWindows()


def toward_citizen(room_col):    # 시민 쪽으로 가
    ser.write(serial.to_bytes([117]))
    global flag
    global room_color
    print(flag, "시민 쪽으로 가")

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    b_lower = np.array([100, 30, 30])
    b_upper = np.array([130, 150, 150])

    r_lower = np.array([150, 120, 30])
    r_upper = np.array([190, 220, 150])

    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        copy = frame.copy()

        if ret:
            convert = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

            if room_color == 'B':
                b_mask = cv.inRange(convert, b_lower, b_upper)
                b_mask = cv.dilate(b_mask, kernel, iterations=1)
                b_mask = cv.GaussianBlur(b_mask, (3, 3), 0)

                cnt = 0
                for i in range(240):
                    for j in range(320):
                        if b_mask[i, j] == 255:
                            cnt += 1
                if cnt > 20000:
                    print(cnt)
                    ser.write(serial.to_bytes([136]))
                    flag = 0
                    break
                cv.imshow("images", b_mask)

            if room_color == 'R':
                r_mask = cv.inRange(convert, r_lower, r_upper)
                r_mask = cv.dilate(r_mask, kernel, iterations=1)
                r_mask = cv.GaussianBlur(r_mask, (3, 3), 0)

                cnt = 0
                for i in range(240):
                    for j in range(320):
                        if r_mask[i, j] == 255:
                            cnt += 1
                if cnt > 20000:
                    print(cnt)
                    ser.write(serial.to_bytes([136]))
                    flag = 0
                    break
                cv.imshow("images", r_mask)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('영상 끝~~~~')
            break

    cv.waitKey()
    cv.destroyAllWindows()

def citizen_transporting(room_col):   # 시민 이송
    global flag
    global room_color
    print(flag, "시민 이송")

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    b_lower = np.array([100, 30, 30])
    b_upper = np.array([130, 150, 150])

    g_lower = np.array([150, 120, 30])
    g_upper = np.array([190, 220, 150])

    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        copy = frame.copy()

        if ret:
            convert = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

            if area == 'B':
                b_mask = cv.inRange(convert, b_lower, b_upper)
                b_mask = cv.dilate(b_mask, kernel, iterations=1)
                b_mask = cv.GaussianBlur(b_mask, (3, 3), 0)

                cnt = 0
                for i in range(240):
                    for j in range(320):
                        if b_mask[i, j] == 255:
                            cnt += 1
                if cnt < 10000:
                    print(cnt)
                    ser.write(serial.to_bytes([137]))
                    flag = 0
                    break
                cv.imshow("images", b_mask)

            if area == 'G':
                g_mask = cv.inRange(convert, g_lower, g_upper)
                g_mask = cv.dilate(g_mask, kernel, iterations=1)
                g_mask = cv.GaussianBlur(g_mask, (3, 3), 0)

                cnt = 0
                for i in range(240):
                    for j in range(320):
                        if g_mask[i, j] == 255:
                            cnt += 1
                if cnt > 20000:
                    print(cnt)
                    ser.write(serial.to_bytes([137]))
                    flag = 0
                    break
                cv.imshow("images", g_mask)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('영상 끝~~~~')
            break

    cv.waitKey()
    cv.destroyAllWindows()


def dangerous_area_print():   # 확진지역 출력
    global flag
    global dangerous
    print(flag, "확진지역 출력")
    print("-------------------------")

    if dangerous[0] == 'A':
        print('A')
        del dangerous[0]
        ser.write(serial.to_bytes([140]))
    elif dangerous[0] == 'B':
        print('B')
        del dangerous[0]
        ser.write(serial.to_bytes([141]))
    elif dangerous[0] == 'C':
        print('C')
        del dangerous[0]
        ser.write(serial.to_bytes([142]))
    elif dangerous[0] == 'D':
        print('D')
        del dangerous[0]
        ser.write(serial.to_bytes([143]))

def toward_room():  # 파지하고 몸체조향
    ser.write(serial.to_bytes([134]))
    global flag
    global area
    print(flag, "파지하고 몸체조향")

    cap = cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    b_lower = np.array([100, 30, 30])
    b_upper = np.array([130, 150, 150])

    g_lower = np.array([150, 120, 30])
    g_upper = np.array([190, 220, 150])

    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        copy = frame.copy()

        if ret:
            convert = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

            if area == 'B':
                b_mask = cv.inRange(convert, b_lower, b_upper)
                b_mask = cv.dilate(b_mask, kernel, iterations=1)
                b_mask = cv.GaussianBlur(b_mask, (3, 3), 0)

                cnt = 0
                for i in range(240):
                    for j in range(320):
                        if b_mask[i, j] == 255:
                            cnt += 1
                if cnt < 10000:
                    print(cnt)
                    ser.write(serial.to_bytes([135]))
                    flag = 11
                    break
                cv.imshow("images", b_mask)

            if area == 'G':
                g_mask = cv.inRange(convert, g_lower, g_upper)
                g_mask = cv.dilate(g_mask, kernel, iterations=1)
                g_mask = cv.GaussianBlur(g_mask, (3, 3), 0)

                cnt = 0
                for i in range(240):
                    for j in range(320):
                        if g_mask[i, j] == 255:
                            cnt += 1
                if cnt > 20000:
                    print(cnt)
                    ser.write(serial.to_bytes([135]))
                    flag = 11
                    break
                cv.imshow("images", g_mask)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('영상 끝~~~~')
            break
    cv.waitKey()
    cv.destroyAllWindows()

def draw_and_compute_lines(edge_img, draw_img, color=(0, 0, 255), thickness=2):
    col_x1, col_y1 = 0, 0
    col_x2, col_y2 = 0, 0
    row_x1, row_y1 = 0, 0
    row_x2, row_y2 = 0, 0
    col_cnt = 0
    row_cnt = 0
    row_deg, col_deg = None, None
    hi = np.zeros((240, 320))

    # min_line_len: 검출할 선분의 취소길이
    # max_line_gap: 직선으로 간주할 최대 에지 점 간격

    rho = 1
    theta = np.pi / 180
    threshold, min_line_len, max_line_gap = 20, 40, 50
    lines = cv.HoughLinesP(edge_img, rho, theta, threshold,
                           minLineLength=min_line_len,
                           maxLineGap=max_line_gap)

    if lines is None:
        print("ERROR: 라인 없음")
        return 0, [0, 0, 0, 0], 0, [0, 0, 0, 0]

    for line in lines:
        for x1, y1, x2, y2 in line:
            # print(x1, y1, x2, y2)
            if (x1 - x2) == 0:
                deg = 90
            else:
                deg = abs(int(math.degrees(math.atan(((y1 - y2) / (x1 - x2))))))
                #print("deg", deg)
            if deg > 60:
                col_cnt += 1
                if y1 >= y2:        # 시작점(x값이 작은 점)이 아래에 있을 때
                    col_x1 += x1
                    col_y1 += y1
                    col_x2 += x2
                    col_y2 += y2
                elif y1 < y2:       # 시작점(x값이 작은 점)이 위에 있을 때
                    col_x1 += x2
                    col_y1 += y2
                    col_x2 += x1
                    col_y2 += y1
            elif deg < 10:
                row_cnt += 1
                row_x1 += x1
                row_y1 += y1
                row_x2 += x2
                row_y2 += y2

            cv.line(hi, (x1, y1), (x2, y2), 255, thickness)
    # cv.imshow('hi', hi)  # 침식 팽창
    # print("------------------------------------")
    if col_cnt > 0:
        col_x1 = int(col_x1 / col_cnt)
        col_y1 = int(col_y1 / col_cnt)
        col_x2 = int(col_x2 / col_cnt)
        col_y2 = int(col_y2 / col_cnt)
        if (col_x1 - col_x2) == 0:
            col_deg = 90
        else:
            col_deg = int(math.degrees(math.atan(((col_y1 - col_y2) / (col_x1 - col_x2)))))
    if row_cnt > 0:
        row_x1 = int(row_x1 / row_cnt)
        row_y1 = int(row_y1 / row_cnt)
        row_x2 = int(row_x2 / row_cnt)
        row_y2 = int(row_y2 / row_cnt)
        if (row_x1 - row_x2) == 0:
            row_deg = 90
        else:
            row_deg = int(math.degrees(math.atan(((row_y1 - row_y2) / (row_x1 - row_x2)))))

    col = [col_x1, col_y1, col_x2, col_y2]
    row = [row_x1, row_y1, row_x2, row_y2]

    cv.line(draw_img, (col_x1, col_y1), (col_x2, col_y2), color, thickness)
    cv.line(draw_img, (row_x1, row_y1), (row_x2, row_y2), color, thickness)
    return row_deg, row, col_deg, col


if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyS0', 4800, timeout=0.0001)
    ser.open
    ser.flush()   # 시리얼 수신 데이터 버리기

    flag = 0
    signal = 0
    color = "black"       # 방위, 시민 인식에 사용하는 색깔
    arrow = " "       # 화살표 방향
    line_flag = 0     # 라인트레이싱 송수신에 이용
    dangerous = []    # 확진구역 방 이름
    area = " "        # 미션구역 색깔
    room_color = " "  # 방 이름 색깔
    room_name = " "   # 방 이름
    mission_flag = 0  # 나가는 시점 확인용 ( = 수행한 미션 개수)
    FPS = 90          # PI CAMERA: 320 x 240 = MAX 90
    flag4_lst = [0, 0, 0, 0]

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
            color = 'yellow'
            flag = 2
        elif signal == 139:
            flag = 12
        elif signal == 128:
            flag = 1
        elif signal == 199:
            flag = 13

        ret, img = cap.read()    # 이미지 받아오기
        
        cv.imshow("img", img)
        #print("main")

        if not ret:
            print("ERROR: 이미지 읽어오기 실패")
            break

        if flag == 0:  continue
        elif flag == 1: line_tracing(img)
        elif flag == 2: line_detecting()
        elif flag == 3: color_detecting(img, color)
        elif flag == 4: direction_recognizing(img)         # 나영
        elif flag == 5: arrow_recognizing()             # 나영
        elif flag == 6: area_recognizing()
        elif flag == 7: room_name_recognizing()         # 나영
        elif flag == 8: room_name_detecting()
        elif flag == 9: citizen_recognizing(room_color)
        elif flag == 10: toward_citizen(room_color)
        elif flag == 11: citizen_transporting(room_color)
        elif flag == 12: dangerous_area_print()
        elif flag == 13: toward_room()
        
        if cv.waitKey(1) & 0xff == ord('q'):  # 'q'누르면 영상 종료
            break
        
    cap.release()
    cv.destroyAllWindows()



