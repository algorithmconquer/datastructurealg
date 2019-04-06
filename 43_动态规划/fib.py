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
	fib = [0] * (n+1)
	fib[0] = 0
	fib[1] = 1
	for i in range(2, n+1):
		fib[i] = fib[i-1] + fib[i-2]
	return fib[n]

print("递归实现结果:"+str(fib_1(0))+","+str(fib_1(1))+","+str(fib_1(5)))
print("递归实现结果:"+str(fib_2(0))+","+str(fib_2(1))+","+str(fib_2(6)))
n = 5
mem = [None] * (n+1)
print("递归实现结果:"+str(fib_mem(n, mem)))
print("递归实现结果:"+str(fib_recurrence(0))+","+str(fib_recurrence(1))+","+str(fib_recurrence(6)))



