import sys
import logging


# 표준 출력을 로그로 보내는 클래스
class LoggerWriter:
	def __init__(self, log_level):
		self.log_level = log_level
		self.buffer = ""
		self.encoding = "utf-8"

	def write(self, message):
		self.buffer += message.replace("\r", "")
		while "\n" in self.buffer:
			pos = self.buffer.index("\n")
			self.log_level(self.buffer[:pos])
			self.buffer = self.buffer[pos + 1:]

	def flush(self):
		if self.buffer:
			self.log_level(self.buffer)
			self.buffer = ""


def set_logger(log_file_path):
	logging.basicConfig(
		level=logging.INFO,
		format="[%(asctime)s](%(levelname)s) %(message)s",
		handlers=[logging.FileHandler(log_file_path, mode="a"), logging.StreamHandler(sys.stdout)],
	)

	# 표준 출력과 오류 출력을 로깅으로 리디렉션
	sys.stdout = LoggerWriter(logging.info)
	sys.stderr = LoggerWriter(logging.error)


if __name__ == "__main__":
	import pandas as pd
	from pathlib import Path
	import tqdm
	import time

	log_file_path = Path(__file__).parent / "데이터_검증_로그.log"
	set_logger(log_file_path)

	# 예제 데이터프레임
	data = {"A": list(range(0, 1000))}
	check_df = pd.DataFrame(data)

	# tqdm을 포함한 반복문
	for idx, row in tqdm.tqdm(check_df.iterrows(), total=len(check_df)):
		a = row["A"]
		time.sleep(0.2)  # 시간 지연
		print(f"Processing A: {a}")

	print("검수 완료!")
