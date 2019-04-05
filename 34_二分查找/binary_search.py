#coding:utf-8

class Solution(object):
	def binarysearch_standard(self, array, target):
		# array = [10, 14, 19, 26, 27, 31, 33, 35, 42, 44]
		# target = 31
		left, right = 0, len(array)-1
		while left <= right:
			mid = int((left + right) / 2) 
			if array[mid] == target:
				return mid
			elif array[mid] > target:
				right = mid - 1
			else:
				left = mid + 1
	# https://leetcode.com/problems/sqrtx/
	def mySqrt(self, x):
		"""
		:type x: int
		:rtype: int
		"""
		if x==0 or x==1:
			return x
		left, right = 1, x
		while left <= right:
			mid = int((left + right) / 2)
			if mid == int(x / mid):
				return mid
			elif mid > int(x / mid):
				right = mid - 1
			else:
				left = mid + 1
				res = mid
		return res
	def mySqrt_newton(self, x):
		res = x
		while res*res > x:
			res = int((res + x / res) / 2)
		return res
	def mySqrt_double(self, x, epision=1e-7):
		left, right = 0, x
		while abs(left-right) > epision:
			mid = (left+right) / 2
			if mid > x / mid:
				right = mid
			else:
				left = mid
		return left
	#https://leetcode.com/problems/valid-perfect-square/
	def isPerfectSquare(self, num):
		"""
		:type num: int
		:rtype: bool
		"""
		left, right = 0, num
		while left <= right:
			mid = int((left + right) / 2)
			if mid * mid == num:
				return True
			elif mid * mid > num:
				right = mid - 1
			else:
				left = mid + 1
		return False



s = Solution()
array = [10, 14, 19, 26, 27, 31, 33, 35, 42, 44]
target = 31
res = s.binarysearch_standard(array, target)
print("res = ", res)
test_seq = [0, 1, 3, 8, 9, 20]
res_seq = [s.mySqrt(i) for i in test_seq]
print("res = ", res_seq)
res_seq = [s.mySqrt_newton(i) for i in test_seq]
print("res = ", res_seq)
res_seq = [s.mySqrt_double(i) for i in test_seq]
print("res = ", res_seq)
res_seq = [s.isPerfectSquare(i) for i in test_seq]
print("res = ", res_seq)


