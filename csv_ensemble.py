from datetime import datetime
import pandas as pd
from tqdm import tqdm
import os
import argparse
from argparse import Namespace, ArgumentParser


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(nargs='+', help='Ex)python csv_ensemble.py result1.csv result2.csv result3.csv', dest='csvs')
    parser.add_argument('--result_path', type=str, default='.local/ensemble')
    parser.add_argument('--result_name', type=str, default='{time}_csv_ensemble.csv')
    
    args = parser.parse_args()
    return args

def main(args:Namespace) -> str:

    now = datetime.now()
    time = now.strftime('%Y-%m-%d %H.%M.%S')
    output_list = args.csvs
    os.makedirs(args.result_path, exist_ok=True)
    result_path = os.path.join(args.result_path, args.result_name.format(time=time))
    # pandas dataframe으로 만들어줍니다.
    df_list = []
    for output in output_list:
        df_list.append(pd.read_csv(output))

    # submission dataframe
    submission = pd.DataFrame()
    submission['image_id'] = df_list[0]['image_id']

    # pixel-wise hard voting 진행
    PredictionString = []

    for idx in tqdm(range(len(df_list[0]))):
        # 각 모델이 뽑은 pixel 넣을 리스트
        pixel_list = []

        for i in range(len(df_list)):
            pixel_list.append(df_list[i]['PredictionString'][idx].split(' '))

        result = ''

        for i in range(len(pixel_list[0])):
            pixel_count = {'0' : 0, '1' : 0, '2' : 0, 
                          '3' : 0, '4' : 0, '5' : 0,
                          '6' : 0, '7' : 0, '8' : 0,
                          '9' : 0, '10' : 0}

            # 각 모델이 뽑은 pixel count
            for j in range(len(pixel_list)):
                pixel_count[pixel_list[j][i]] += 1

            # 제일 많이 vote된 pixel 값
            voted_pixel = [key for key, value in pixel_count.items() if value == max(pixel_count.values())]

            # voted_pixel이 1개인 경우
            if len(voted_pixel) == 1:
                result += voted_pixel[0] + ' '
            # 동점이 나온 경우
            else:
                # 성능이 좋았던 모델부터 값이 voted_pixel에 있다면 result로 고르기
                for j in range(len(pixel_list)):
                    pixel_candidate = pixel_list[j][i]

                    if pixel_candidate in voted_pixel:
                        result += pixel_candidate + ' '
                        break
                    
        # 마지막 공백 제거
        result = result[:-1]

        PredictionString.append(result)

    # submission csv 만들기
    submission['PredictionString'] = PredictionString
    submission.to_csv(result_path , index=False)
    print(f"The result has been saved at '{result_path}'")
    return result_path

if __name__ == "__main__":
    args = parse_args()
    main(args)