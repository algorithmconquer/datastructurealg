#coding:utf-8
# 递归实现:0,1,1,2,3,5,8,13,21,...
def fib_1(n):
	if n<=0:
		return 0
	elif n==1:
		return 1
	else:
		return fib_1(n-1) + fib_1(n-2)
def fib_2(n):
	return  fib_2(n-1) + fib_2(n-2) if n>=2 else n
# 递归+记忆化
def fib_mem(n, mem):
	if n<=0:
		return 0
	elif n==1:
		return 1
	elif mem[n]==None:
		mem[n] = fib_mem(n-1, mem) + fib_mem(n-2, mem)
	return mem[n]
# 递推
def fib_recurrence(n):
	if n<=1:
		return n
	fib = [0] * n
	fib[0] = 1
	fib[1] = 1
	for i in range(2, n):
		fib[i] = fib[i-1] + fib[i-2]
	return fib[n-1]
import numpy as np
def count_parhs(array):
	for i in range(1, 8):
		for j in range(1, 8):
			if int(array[i, j])==100:
				array[i, j] = 0
			else:
				array[i, j] = array[i-1, j] + array[i, j-1]
	return array[7, 7]

class Solution(object):
	def climbStairs(self, n):
		"""
		:type n: int
		:rtype: int
		"""
		if n<= 2:
			return n
		fib = [0] * n
		fib[0], fib[1] = 1, 2
		for i in range(2, n):
			fib[i] = fib[i-1] + fib[i-2]
		return fib[n-1]

	def climbStairs2(self, n):
		"""
		:type n: int
		:rtype: int
		"""
		if n<=2:
			return n
		fib1 = 1
		fib2 = 2
		all_ways = 0
		for i in range(2, n):
			all_ways = fib1 + fib2
			fib1 = fib2 
			fib2 = all_ways
		return all_ways

	def climbStairs3(self, n):
		"""
		:type n: int
		:rtype: int
		"""
		x, y = 1, 1
		for _ in range(1, n):
			x, y= y, x+y
		return y
	# 120:https://leetcode.com/problems/triangle/description/
# 	[
#   [2],
#   [3,4],
#   [6,5,7],
#   [4,1,8,3]
# ]
	def minimumTotal(self, triangle):
		"""
		:type triangle: List[List[int]]
		:rtype: int
		"""
		# 状态定义:dp[i, j]--从最下面走到i, j这个点路径和的最小值；
		# 状态方程:dp[i, j]=min(dp[i+1, j], dp[i+1, j+1]) + a[i, j]
		rows = len(triangle)
		for i in range(rows-2, -1, -1):
			for j in range(i+1):
				#dp[i, j] = min(dp[i+1, j], dp[i+1, j+1]) + triangle[i][j]
				triangle[i][j] = min(triangle[i+1][j], triangle[i+1][j+1]) + triangle[i][j]
		return triangle[0][0]
	def minimumTotal2(self, triangle):
		rows = len(triangle)
		mini = triangle[rows-1]
		for i in range(rows-2, -1, -1):
			for j in range(len(triangle[i])):
				#triangle[i][j] = min(triangle[i+1][j], triangle[i+1][j+1]) + triangle[i][j]
				mini[j] = min(mini[j], mini[j+1]) + triangle[i][j]
		return mini[0]




print("递归实现结果:"+str(fib_1(0))+","+str(fib_1(1))+","+str(fib_1(5)))
print("递归实现结果:"+str(fib_2(0))+","+str(fib_2(1))+","+str(fib_2(6)))
n = 5
mem = [None] * (n+1)
print("递归实现结果:"+str(fib_mem(n, mem)))
print("递归实现结果:"+str(fib_recurrence(0))+","+str(fib_recurrence(1))+","+str(fib_recurrence(6)))

array = np.zeros((8, 8))
array[0], array[:, 0] = 1, 1
array[1, 2], array[1, 6]= 100,  100
array[2, 1], array[2, 3], array[2, 4] = 100,  100, 100
array[3, 5] = 100
array[4, 2], array[4, 5], array[4, 7] = 100,  100, 100
array[5, 3] = 100
array[6, 1], array[6, 5]= 100,  100
print("Init array =", array)
print(count_parhs(array))
print("Res array =", array)

s = Solution()
print(s.climbStairs(1), s.climbStairs(2), s.climbStairs(3), s.climbStairs(4), s.climbStairs(5))
print(s.climbStairs2(1), s.climbStairs2(2), s.climbStairs2(3), s.climbStairs2(4), s.climbStairs2(5))
print(s.climbStairs3(1), s.climbStairs3(2), s.climbStairs3(3), s.climbStairs3(4), s.climbStairs3(5))

triangle = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
#print(s.minimumTotal(triangle))
print(s.minimumTotal2(triangle))







