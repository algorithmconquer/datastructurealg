# 一、原理

69， 35，34，74，153，33，278，162，287，275

A1                  A2

|----------------｜------------------------------------------|

|----------------------------------｜------------------------|

​                                            B1                             B2

对于区间[A1, B2];

假设有一个总区间，经由我们的check函数判断后，可分成两部分，
这边以o作true，v作false示意较好识别:

如果我们的目标是下面这个v，那么就必须使用模板1

................vooooooooo

假设经由check划分后，整个区间的属性与目标v如下，则我们必须使用模板2

oooooooov...................

模板1就是在满足chek()的区间内找到左边界，模板2在满足check()的区间内找到右边界。然后无论是左边界还是右边界，都应该是整个区间中某一段满足某性质（如单调不降）与另一段不满足该性质的分界点（也就是同学的v）。

https://leetcode-cn.com/problems/search-insert-position/solution/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/;

```shell
分析题意，挖掘题目中隐含的 单调性；
1.while (left < right) 退出循环的时候有left == right 成立，因此无需考虑返回 left 还是 right；
2.始终思考下一轮搜索区间是什么，如果是 [mid, right] 就对应 left = mid ，如果是 [left, mid - 1] 就对应 right = mid - 1，是保留 mid 还是+1、-1就在这样的思考中完成；
3.从一个元素什么时候不是解开始考虑下一轮搜索区间是什么 ，把区间分为2个部分（一个部分肯定不存在目标元素，另一个部分有可能存在目标元素），问题会变得简单很多，这是一条非常有用的经验；
4.每一轮区间被划分成2部分，理解"区间划分"决定中间数取法（无需记忆，需要练习+理解 ），在调试的过程中理解区间和中间数划分的配对关系：
区间划分[left, mid]与[mid+1, right] ，"mid被分到左边"，对应int mid = left + (right - left)/2,计算mid时不需要+1;
区间划分[left, mid-1]与[mid, right] ，"mid被分到右边"，对应int mid = left + (right - left + 1)/2,计算mid时需要+1。
5.退出循环的时候有left ==right成立，此时如果能确定问题一定有解，返回left即可，如果不能确定，需要单独判断一次。
```

**二分查找的两种思路（请特别留意第 2 种思路，掌握它能思路清晰地解决「力扣」上的所有二分查找问题）**

```shell
思路1：在循环体内部查找元素
while(left <= right) 这种写法表示在循环体内部直接查找元素；
退出循环的时候left 和 right 不重合，区间 [left, right] 是空区间。
思路 2：在循环体内部排除元素（重点）
while(left < right) 这种写法表示在循环体内部排除元素；
退出循环的时候left和right重合，区间[left, right]只剩下成1个元素，这个元素有可能就是我们要找的元素。
第2种思路可以归纳为「左右边界向中间走，两边夹」，这种思路在解决复杂问题的时候，可以使得思考的过程变得简单。
```

## 1.1 二分模板1

二分右边这个端点(B1)；如果中点M在B1,B2之间(也就是M > target)；则需要找的分界点在M左边；

```shell
当我们将区间[l, r]划分成[l, mid]和[mid + 1, r]时，其更新操作是r = mid或者l = mid + 1;计算mid时不需要加1。
```

## 1.2 二分模板2

二分左边这个端点(A2)；如果中点M在A1,A2之间(也就是M < target)；

```shell
当我们将区间[l, r]划分成[l, mid - 1]和[mid, r]时，其更新操作是r = mid - 1或者l = mid;，此时为了防止死循环，计算mid时需要加1。
```

# 二、题目

## 2.1 x的平方根_69

```shell
示例 2:
输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。

```

代码：

```c++
class Solution {
public:
    int mySqrt(int x) {
        int l=0, r=x;
        while(l<r){
            int mid=l+(long long)r+1>>1;
            // 严格>target的元素一定不是解
            if(mid>x/mid) r=mid-1;//下一轮搜索区间是 [l, mid-1]
            else l=mid;
        }
        return l;
    }
};
class Solution {
public:
    int mySqrt(int x) {
        int l = 0, r = x;
        while(l < r){
            int mid = l + (long long)r + 1 >> 1; // 计算mid时需要加1
            if(mid <= x / mid){// oooooooov
                l = mid;
            }else{
                r = mid - 1;
            }
        }
        return l;// or r
    }
};
```

## 2.2 搜索插入位置_35

```shell
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。
示例 1:
输入: [1,3,5,6], 5
输出: 2 // mid>=target的最小值,res=1;mid<=target的最大值,res=0;
示例 2:
输入: [1,3,5,6], 2
输出: 1;//mid>=target的最小值 or mid<=target的最大值
示例 3:
输入: [1,3,5,6], 7
输出: 4
```

代码:

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        if(nums.empty() || nums.back()<target) return nums.size();
        int n=nums.size();
        int l=0, r=n-1;
        while(l<r){
            int mid=l+r>>1;
            if(nums[mid] >= target) r=mid;
            else l=mid+1;
        }
        return r;
    }
};
```

## 2.3 在排序数组中查找元素的第一个和最后一个位置_34

```shell
示例 1：
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
示例 2：
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
示例 3：
输入：nums = [], target = 0
输出：[-1,-1]
```

代码：

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.empty()) return {-1, -1};
        int n = nums.size();
        int start=0, end=0;
        int l=0, r=n-1;
        while(l<r){
            int mid=l+r>>1;
            // 严格小于target的元素一定不是解
            if(nums[mid]<target) l=mid+1;//下一次搜索区间[mid+1, r]
            else r=mid;
        }
        start = r;
        if(target != nums[start]) return {-1, -1};
        l=0, r=n-1;
        while(l<r){
            int mid=l+r+1>>1;
            if(nums[mid]>target) r=mid-1;//下一次搜索区间[l, mid-1]
            else l=mid;
        }
        end = r;
        return {start, end};
    }
};
```

## 2.4 搜索二维矩阵_74

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

```shell
示例 1
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true
示例 2
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
输出：false
```

 代码：

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int l=0,r=m*n-1;
        while(l<r){
            int mid=l+r+1>>1;
            // 严格>于target的元素一定不是解
            if(matrix[mid/n][mid%n]>target) r = mid-1;
            else l=mid;
        }
        return matrix[r/n][r%n]==target?true:false;
    }
};
```

## 2.5 寻找旋转排序数组中的最小值_153

假设按照升序排序的数组在预先未知的某个点上进行了旋转。例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] 。
请找出其中最小的元素。

```shell
示例 1：
输入：nums = [3,4,5,1,2]==>隐藏条件:nums.back()<nums[0];
输出：1
示例 2：
输入：nums = [4,5,6,7,0,1,2]
输出：0
示例 3：
输入：nums = [1]
输出：1
```

代码

```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
      if(nums.back() > nums[0]) return nums[0];
      int n = nums.size();
      int l=0, r=n-1;
      while(l<r){
        int mid=l+r>>1;
        // 严格小于target的元素一定不是解
        if(nums[mid] > nums.back()) l=mid+1;
        else r=mid;
      }
      return nums[l];
    }
};
```

## 2.6 搜索旋转排序数组_33

```shell
示例 1：
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
示例 2：
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
示例 3：
输入：nums = [1], target = 0
输出：-1
```

代码：

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
      int n = nums.size();
      int l = 0, r = n-1;
      // 计算min_v
      while(l<r){
        int mid = l+r>>1;
        if(nums[mid] > nums.back()) l = mid + 1;
        else r = mid;
      }
      if(target > nums.back()) l = 0, r = r - 1;
      else l = r, r = nums.size() - 1;
      while(l<r){
        int mid = l+r+1>>1;
        // 严格target的元素一定不是解
        if(nums[mid]>target) r=mid-1;
        else l=mid;
      }
      return nums[l]==target ? l : -1 ;
    }
};
```

## 2.7 寻找峰值_162

峰值元素是指其值大于左右相邻值的元素。

给你一个输入数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞ 。

```shell
示例 1：
输入：nums = [1,2,3,1]
输出：2
解释：3是峰值元素，你的函数应该返回其索引2。

示例 2：
输入：nums = [1,2,1,3,5,6,4]
输出：1或5 
解释：你的函数可以返回索引1，其峰值元素为2；
     或者返回索引5， 其峰值元素为6。
```



