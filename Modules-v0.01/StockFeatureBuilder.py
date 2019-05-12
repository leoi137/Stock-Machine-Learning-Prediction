import numpy as np
import pandas as pd
import time


class FeatureData():

	def __init__(self, data, symbols, kind, future):

		self.data = data
		self.symbols = symbols
		self.kind = kind
		self.future = future

		self.original_columns = set()

		self.start_bars = 0
		self.prev_bars = 30
		self.period = 1

	def get_original_columns(self, data):

		for s in self.symbols:
			for col in data[s].columns:
				self.original_columns.add(col)

	def main(self, combine):

		print("Creating Features...\n")
		start = time.perf_counter()

		self.get_original_columns(self.data)

		self.bar_data = self.change_type(self.data)
		self.bar_data = self.create_bars(self.bar_data)
		self.bar_data = self.perc_change(self.bar_data)
		self.bar_data = self.pct_change_derivative(self.bar_data)
		self.bar_data = self.derivative(self.bar_data)
		self.bar_data = self.mult_stdv(self.bar_data)
		self.bar_data = self.mult_stdv_derivative(self.bar_data)
		self.bar_data = self.add_ema(self.bar_data)
		self.bar_data = self.price_speed(self.bar_data)
		self.bar_data = self.price_speed_derivative(self.bar_data)
		self.bar_data = self.ema_price_difference(self.bar_data)
		self.bar_data = self.pin_bar(self.bar_data)
		self.bar_data = self.bar_differences(self.bar_data)
		self.bar_data = self.bar_differences_derivative(self.bar_data)
		self.bar_data = self.bar_perc(self.bar_data)

		
		print("Creating the features took: {:0.4f} seconds\n".format(time.perf_counter() - start))
		
		self.bar_data = self.create_target(self.bar_data)
		
		print("Cleaning and formating the data...")
		start = time.perf_counter()
		self.bar_data = self.clean_data(self.bar_data, combine)
		print("Clearning the data took: {:0.4f} seconds".format(time.perf_counter() - start))

		return self.bar_data

	def change_type(self, data):

		print("Changing data type from 64-bit to 32-bit...")

		start = time.perf_counter()

		for s in self.symbols:
			data[s] = data[s].astype('float32')

		print("Type conversion took: {:0.4f}\n".format(time.perf_counter() - start))	

		return data

	def create_bars(self, data):

		print("Creating Bars...")
		start = time.perf_counter()

		for s in self.symbols:
			for col in data[s].columns:
				for i in range(self.start_bars, self.prev_bars, self.period):
					data[s]["{0} {1}".format(col, str(i + self.period))] = data[s]['{}'.format(col)].diff(i + self.period)
				for i in range(self.start_bars, self.prev_bars, self.period):
					data[s]["{0} p - {1}".format(col, str(i + self.period))] = data[s]['{}'.format(col)].shift(i + self.period)

		print("Bar creation took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def perc_change(self, data):

		print("Taking percentage change...")
		start = time.perf_counter()

		for s in self.symbols:
			data[s]['pct'] = data[s]['close'].pct_change() * 100
			data[s]['high pct'] = data[s]['high'].pct_change() * 100
			data[s]['low pct'] = data[s]['low'].pct_change() * 100
			data[s]['pct fut'] = ((data[s]['close'].shift(-int(self.future)) - data[s]['close']) / data[s]['close']) * 100
			# data[s]['pct up'] = ((data[s]['high'].shift(-int(self.future)) - data[s]['close']) / data[s]['close']) * 100
			# data[s]['pct down'] = ((data[s]['low'].shift(-int(self.future)) - data[s]['close'])/ data[s]['close']) * 100

		for s in self.symbols:
			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]["pct {}".format(str(i + self.period))] = data[s]['close'].pct_change(i + self.period)

		print("Percentage change took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def pct_change_derivative(self, data):

		print("Taking percentage change...")
		start = time.perf_counter()

		for s in self.symbols:
			data[s]['pct: derivative'] = np.gradient(data[s]['pct'])
			data[s]['high pct: derivative'] = np.gradient(data[s]['high pct'])
			data[s]['low pct: derivative'] = np.gradient(data[s]['low pct'])

		print("Percentage change took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def derivative(self, data):

		print("Taking Derivatives")
		start = time.perf_counter()

		for s in self.symbols:
			for col in self.original_columns:
				data[s][f'{col} - derivative'] = np.gradient(data[s][str(col)])

		for s in self.symbols:
			for col in self.original_columns:
				for i in range(self.start_bars, 10, self.period):
					data[s][f'{col} - derivative: {str(i + self.period)}'] = data[s][f'{col} - derivative'].shift(i + self.period)

		print("Derivatives took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def mult_stdv(self, data, rolling = [5, 10]):

		print("Taking Multiple STDV...")
		start = time.perf_counter()

		for s in self.symbols:
			for i in range(self.start_bars, 10, self.period):
				for r in rolling:
					data[s]["close STDV {}: {}".format(r, str(i + self.period))] = data[s]['close'].shift(i + self.period).rolling(r).std()
					data[s]["volume STDV {}: {}".format(r, str(i + self.period))] = data[s]['volume'].shift(i + self.period).rolling(r).std()
					#data[s]["high STDV {}: {}".format(r, str(i + self.period))] = data[s]['high'].shift(i + self.period).rolling(r).std()
					#data[s]["low STDV {}: {}".format(r, str(i + self.period))] = data[s]['low'].shift(i + self.period).rolling(r).std()
					#data[s]["open STDV {}: {}".format(r, str(i + self.period))] = data[s]['open'].shift(i + self.period).rolling(r).std()

		print("Multiple STDV took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def mult_stdv_derivative(self, data, rolling = [5, 10]):

		print("Taking Multiple STDV Derivative...")
		start = time.perf_counter()

		for s in self.symbols:
			for i in range(self.start_bars, 10, self.period):
				for r in rolling:
					data[s]["close STDV {}: {} derivative".format(r, str(i + self.period))] = np.gradient(
						data[s]["close STDV {}: {}".format(r, str(i + self.period))])


		print("Multiple Derivative STDV took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def add_ema(self, data):

		print("Calculating EMAs...")
		start = time.perf_counter()

		minEMA = 0
		maxEMA = 30
		periodEMA = 1
		EMA = [8, 21]

		for e in EMA:
			for s in self.symbols:
				data[s]['EMA {}'.format(str(e))] = data[s]['close'].ewm(e).mean()
				for i in range(minEMA, maxEMA, periodEMA):
					data[s]['EMA {0}: {1}'.format(str(e), 
						str(i + periodEMA))] = data[s]['EMA {}'.format(e)].shift(i + periodEMA)
		print("EMAs took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def ema_price_difference(self, data, EMA = [8, 21]):

		print("Calculating EMAs Differences...")
		start = time.perf_counter()

		minEMA = 0
		maxEMA = 30
		periodEMA = 1

		for s in self.symbols:
			data[s][f'EMA {EMA[0]} - {EMA[1]}'] = data[s]['close'].ewm(EMA[0]).mean() - data[s]['close'].ewm(EMA[1]).mean()
			for i in range(minEMA, maxEMA, periodEMA):
				data[s][f'EMA {EMA[0]} - {EMA[1]}: {i + periodEMA}'] = data[s][f'EMA {EMA[0]} - {EMA[1]}'].shift(i + periodEMA)

		print("EMAs Differences took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def pin_bar(self, data):

		print("Creating Pin bars...")
		start = time.perf_counter()

		# prev_pins = 10
		# pin_period = 1

		for s in self.symbols:

			data[s]['diff'] = (data[s]['high'] - data[s]['low']).abs()
			data[s]['std'] = data[s]['diff'].rolling(20).std()

			bull_bullcpin = (((data[s]['open'] - data[s]['low'])/(
				data[s]['high'] - data[s]['low'])) > 0.618).astype(int)
			bull_bearcpin = (((data[s]['close'] - data[s]['low'])/(
				data[s]['high'] - data[s]['low'])) > 0.618).astype(int)
			data[s]['Up pin_bar'] = ((bull_bullcpin & bull_bearcpin & data[s]['diff'] > data[s]['std'])).astype(int)

			bear_bullcpin = ((data[s]['high'] - data[s]['open'])/(
				data[s]['high'] - data[s]['low'])) > 0.618
			bear_bearcpin = ((data[s]['high'] - data[s]['close'])/(
				data[s]['high'] - data[s]['low'])) > 0.618
			data[s]['Down pin_bar'] = (bear_bullcpin & bear_bearcpin & (data[s]['diff'] > data[s]['std'])).astype(int)

			data[s].drop(['diff', 'std'], axis = 1, inplace = True)

			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]['Down pin_bar {}'.format(i + self.period)] = data[s]['Down pin_bar'].shift(i + self.period)
				data[s]['Up pin_bar {}'.format(i + self.period)] = data[s]['Up pin_bar'].shift(i + self.period)

		print("Pin Bar took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def bar_perc(self, data):

		print("Getting bar percentages...")

		start = time.perf_counter()
		for s in self.symbols:
			data[s]['body'] = ((data[s]['close'] - data[s]['open']) / (data[s]['high'] - data[s]['low'])).abs()
			data[s]['Up'] = ((data[s]['close'] - data[s]['open']) > 0).astype(int)
			for i in range(0, self.prev_bars, self.period):
				data[s]['body {}'.format(i + self.period)] = data[s]['body'].shift(i + self.period)
				data[s]['Up {}'.format(i + self.period)] = data[s]['Up'].shift(i + self.period)


		print("Bar percentages took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def price_speed(self, data):

		print("Getting speed of bars...")
		start = time.perf_counter()

		for s in self.symbols:
			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]['Speed / {}bars'.format(i + self.period)] = (data[s]['close'] - data[s]['close'].shift(i + self.period))/(i + self.period)

		print("Speed took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def price_speed_derivative(self, data):

		print("Taking derivative of speed of bars...")
		start = time.perf_counter()

		p_period = 2
		for s in self.symbols:
			for i in range(self.start_bars, self.prev_bars, p_period):
				data[s]['Speed / {}bars: derivative'.format(i + p_period)] = np.gradient(data[s]['Speed / {}bars'.format(i + p_period)])

		print("Derivative of Speed took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def bar_differences(self, data):

		print("Taking bar ohlc differences...")
		start = time.perf_counter()

		for s in self.symbols:
			data[s]['high - low'] = data[s]['high'] - data[s]['low']
			data[s]['close - open'] = data[s]['close'] - data[s]['open']
			data[s]['high - open'] = data[s]['high'] - data[s]['open']
			data[s]['low - open'] = data[s]['low'] - data[s]['open']
			data[s]['high - close'] = data[s]['high'] - data[s]['close']
			data[s]['low - close'] = data[s]['low'] - data[s]['close']

			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]['high - low: {}'.format(i + self.period)] = data[s]['high - low'].shift(i + self.period)
				data[s]['close - open: {}'.format(i + self.period)] = data[s]['close - open'].shift(i + self.period)
				data[s]['high - open: {}'.format(i + self.period)] = data[s]['high - open'].shift(i + self.period)
				data[s]['low - open: {}'.format(i + self.period)] = data[s]['low - open'].shift(i + self.period)
				data[s]['high - close: {}'.format(i + self.period)] = data[s]['high - close'].shift(i + self.period)
				data[s]['low - close: {}'.format(i + self.period)] = data[s]['low - close'].shift(i + self.period)

		print("Bar Diff took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def bar_differences_derivative(self, data):

		print("Taking derivative of bar ohlc differences...")
		start = time.perf_counter()

		for s in self.symbols:
			data[s]['high - low: derivative'] = np.gradient(data[s]['high - low'])
			data[s]['close - open: derivative'] = np.gradient(data[s]['close - open'])
			data[s]['high - open: derivative'] = np.gradient(data[s]['high - open'])
			data[s]['low - open: derivative'] = np.gradient(data[s]['low - open'])
			data[s]['high - close: derivative'] = np.gradient(data[s]['high - close'])
			data[s]['low - close: derivative'] = np.gradient(data[s]['low - close'])

		print("Derivative of Bar Diff took: {:0.4f}\n".format(time.perf_counter() - start))
		
		return data

	def create_target(self, data):
		# It is likely better to combine the basic features first rather than after getting new features.

		print("Creating Target...\n")

		if self.kind == 'Regression':
			for s in self.symbols:
				data[s]['Target'] = data[s]['close'].shift(-1)
				#data[s]['Target'] = data[s]['close'].shift(-1).pct_change()

		if self.kind == 'Classification':
			sym_dict = {}

			for s in self.symbols:
				self.start_time = time.perf_counter()
				print("Loading {}...".format(str(s).upper()))
				sym_dict[s] = []

				for i in range(0, len(data[s])):

					# if data[s]['pct up'][i] > 3:
					# 	sym_dict[s].append([1])
					# else:
					# 	sym_dict[s].append([0])

					if data[s]['pct fut'][i] > 2.5:
						sym_dict[s].append([1])
					elif data[s]['pct fut'][i] < -2.5:
						sym_dict[s].append([-1])
					else:
						sym_dict[s].append([0])

					# if (data[s]['pct up'][i] > 2.75) and (data[s]['pct down'][i] < -2.75):
					# 	sym_dict[s].append([0])
					# else:
					# 	if data[s]['pct up'][i] > 2.75:
					# 		sym_dict[s].append([1])
					# 	elif data[s]['pct down'][i] < -2.75:
					# 		sym_dict[s].append([-1])
					# 	else:
					# 		sym_dict[s].append([0])

				# print(f'Lengths Equal: {len(sym_dict[s]) == len(data[s])}')

				print("* {0} Completed in {1:0.4f} seconds\n".format(str(s).upper(), time.perf_counter() - self.start_time))

			self.start_time = time.perf_counter()
			print("Concatinating Targets...".format())
			for s in self.symbols:        
				data[s] = pd.concat([data[s], pd.DataFrame(sym_dict[s], index = data[s].index, columns = ['Target'])], axis = 1)

			print("Target concatination completed in {:0.4f} seconds\n".format(time.perf_counter() - self.start_time))

		return data

	def clean_data(self, data, combine):

		target_list = []
		for s in self.symbols:
		    try:
		        # data[s].drop(['pct up', 'pct down'], axis = 1, inplace = True)
		        data[s].drop(['pct fut'], axis = 1, inplace = True)
		        # print("Dropped pct up and pct down")
		    except KeyError:
		        pass
		            
		for s in self.symbols:
		 	data[s].dropna(inplace = True)
		 	data[s] = data[s].replace([np.inf, -np.inf], np.nan)
		 	data[s].ffill(inplace = True)

		# print("Dropping...")
		# for s in self.symbols:
		# 	print(f"Symbol: {s}")
		# 	target = data[s]['Target'][data[s]['Target'] == 0]
		# 	data[s] = data[s].drop(target.sample(frac = 0.97).index, axis = 0)

		if combine:
			all_data = []
			for k in data.keys():
				all_data.append(data[k])
			data = pd.concat(all_data).dropna()

		return data