import ast

def prepare_data_for_validation(received_data):
    try:
        # command 추출
        command = received_data[0].strip()

        # data 추출 및 불필요한 문자열 제거
        data = [item.strip("' ") for item in received_data[1:]]
        data = [item for item in data if item]  # 빈 문자열 제거
        data = [item.strip("']") for item in data]
        data = [item for item in data if item != "["]

        # data_str 형식으로 데이터 가공
        if len(data) == 1:
            data_str = data[0]
        else:
            data_str = data

        # parameter_count 계산
        if isinstance(data_str, list):
            parameter_count = len(data_str)
        else:
            parameter_count = 1

        return command, data_str, parameter_count

    except Exception as e:
        return None, str(e)


if __name__ == '__main__':
    received_data_1 = ['333 ', '화가 난다']
    received_data_2 = ['555', " ['", "이거'", " '", "요고'", " '", "저거']"]

    # prepare_data_for_validation 함수를 호출하여 결과 출력
    command_1, data_str_1, parameter_count_1 = prepare_data_for_validation(received_data_1)
    command_2, data_str_2, parameter_count_2 = prepare_data_for_validation(received_data_2)

    # 결과 출력
    print("Case 1 - Command:", command_1)
    print("Case 1 - Data:", data_str_1)
    print("Case 1 - Parameter Count:", parameter_count_1)

    print("Case 2 - Command:", command_2)
    print("Case 2 - Data:", data_str_2)
    print("Case 2 - Parameter Count:", parameter_count_2)