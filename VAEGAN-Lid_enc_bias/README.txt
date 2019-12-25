1. 필요한 파이썬 라이브러리 설치(tensorflow, librosa 등등)

2. data/speakers 에 training할 데이터 저장 / data/speakers_test 에 test 데이터 저장

3. preprocess.py 실행 (processed 폴더에 preprocess한 파일들이 화자마다 생성됨)

4. python3 main-vawgan.py 실행

5. convert.py 로 음성 변환 (학습 중간에 실행 가능)

Ex) python3 convert.py --source_speaker SF1 --target_speaker TM3 --model_dir ./logdir/train/all_model_v2/model.ckpt-XXXX
