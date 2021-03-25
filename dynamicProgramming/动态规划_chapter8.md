# 一、原理

参考：https://zhuanlan.zhihu.com/p/91582909；

动态规划，无非就是利用**历史记录**，来避免重复计算。而这些**历史记录**，得需要一些**变量**来保存，一般是用**一维数组**或者**二维数组**来保存。下面先来看看做动态规划题很重要的三个步骤 :

**第一步骤**：定义**数组元素dp[i]的含义**，上面说了，会用一个数组来保存历史记录，假设用一维数组$dp[i]$吧。这个时候有一个非常非常重要的点，就是规定你这个数组元素的含义。==>**状态定义**

**第二步骤**：找出**数组元素之间的关系式**，动态规划有一点类似于我们高中学习时的**归纳法**，当要计算$dp[n]$时，是可以利用$dp[n-1],dp[n-2].....dp[1]$来推出 dp[n] 的，也就是可以利用**历史数据**来推出新的元素值，所以要找出数组元素之间的关系式，例如 dp[n] = dp[n-1] + dp[n-2]，这个就是他们的关系式了。==>**状态转移方程**

**第三步骤**：找出**初始值**。学过**数学归纳法**的都知道，虽然知道了数组元素之间的关系式，例如 dp[n] = dp[n-1] + dp[n-2]，可以通过 dp[n-1] 和 dp[n-2] 来计算dp[n]，但是，得知道初始值啊，例如一直推下去的话，会由 dp[3] = dp[2] + dp[1]。而 dp[2] 和 dp[1] 是不能再分解的了，所以我们必须要能够直接获得dp[2]和dp[1]的值，而这，就是**所谓的初始值**。==>**初始值**

# 二、题目

## 2.1 编辑距离_leetcode72

```shell
示例 1：
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
示例 2：
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

代码:

```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        vector<vector<int>> f(m+1, vector<int>(n+1));
        for(int i=0; i<=m; i++) f[i][0] = i;
        for(int j=0; j<=n; j++) f[0][j] = j;
        for(int i=1; i<=m; i++){
            for(int j=1; j<=n; j++){
                // 插入，删除
                f[i][j] = min(f[i][j-1], f[i-1][j])+1;
                // 替换
                f[i][j] = min(f[i-1][j-1]+(word1[i-1]!=word2[j-1]), f[i][j]);
            }
        }
        return f[m][n];
    }
};
// 优化:原来:空间复杂度为O(n*m);
// 如果要计算第i行的值，我们最多只依赖第i-1行的值，不需要用到第i-2行及其以前的值，所以一样可以采用一维dp来处理的。
// 但是还要保存f[i-1][j-1];因为f[i][j]依赖f[i][j-1], f[i-1][j],f[i-1][j-1] 
class Solution {
public:
    int minDistance(string s, string t) {
        int m = s.size(), n = t.size();
        vector<int> f(n+1);
        for(int i=0; i<=n; i++) f[i] = i;
        for(int i=1; i<=m; i++){
            int temp = f[0];// 
            f[0] = i;
            for(int j=1; j<=n; j++){
                int pre = temp;// pre 相当于之前的 dp[i-1][j-1]
                temp = f[j];
                // 如果 word1[i] 与 word2[j] 相等。第 i 个字符对应下标是 i-1
                if (s[i-1] == t[j-1]) f[j] = pre;
                else f[j] = min(min(f[j], f[j-1]), pre) + 1;
            }
        }
        return f[n];
    }
};
```

例如:

```shell
   i     j
abcd,abcde;==>add操作:f[i][j-1]+1
插入操作(说明s短)；
    i    j
abcde,abcd;==>delete操作:f[i-1][j]+1
删除操作(说明s长)；
```

见下图

<img src="/Users/zenmen/Projects/datastructurealg/images/image-20210218115503915.png" alt="image-20210218115503915" style="zoom:50%;" />

## 2.2 最长递增子序列_leetcode300

```shell
示例 1：
输入：nums = [10,9,2,5,3,7,101,18]
输出：4

解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
示例 2：
输入：nums = [0,1,0,3,2,3]
输出：4

示例 3：
输入：nums = [7,7,7,7,7,7,7]
输出：1
```

状态定义，状态转移方程；

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> f(n); // 什么时候需要申请数组长度为n+1?
        // 在for循环外,没有初始化f[i],则需要在for循环里进行初始化;
        for(int i=0; i<n; i++){
            f[i] = 1;
            for(int j=0; j<i; j++){
                if(nums[j]<nums[i])
                f[i] = max(f[i], f[j]+1);
            }
        }
        int res = INT_MIN;
        for(int i=0; i<n; i++) res = max(res, f[i]);
        return res;
    }
};
```

## 2.3 三角形最小路径和_leetcode120

给定一个三角形triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点在这里指的是下标与上一层结点下标 相同或者等于上一层结点下标+1的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

```shell
示例 1：
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：如下面简图所示：
   2
  3 4
 6 5 7
4 1 8 3
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。

示例 2：
输入：triangle = [[-10]]
输出：-10
```

代码:

```c++
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int m = triangle.size();
        vector<vector<int>> f(m, vector<int>(m));
        f[0][0] = triangle[0][0];
        for(int i=1; i<m; i++){
            for(int j=0; j<=i; j++){
                f[i][j] = INT_MAX;
                if (j==0) f[i][j] = min(f[i][j], f[i-1][j] + triangle[i][j]);
                else if (i==j) f[i][j] = min(f[i][j], f[i-1][j-1] + triangle[i][j]);
                else f[i][j] = min(f[i-1][j-1], f[i-1][j]) + triangle[i][j];
            }
        }
        int res = INT_MAX;
        for(int i=0; i<m; i++) res = min(res, f[m-1][i]);
        return res;
    }
};
// 进一步优化空间复杂度:
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int m = triangle.size();
        vector<vector<int>> f(2, vector<int>(m));
        f[0][0] = triangle[0][0];
        for(int i=1; i<m; i++){
            for(int j=0; j<=i; j++){
                f[i&1][j] = INT_MAX;
                if (j==0) f[i&1][j] = min(f[i&1][j], f[i-1&1][j] + triangle[i][j]);
                else if (i==j) f[i&1][j] = min(f[i&1][j], f[i-1&1][j-1] + triangle[i][j]);
                else f[i&1][j] = min(f[i-1&1][j-1], f[i-1&1][j]) + triangle[i][j];
            }
        }
        int res = INT_MAX;
        for(int i=0; i<m; i++) res = min(res, f[m-1&1][i]);
        return res;
    }
};
```

## 2.4 零钱兑换_leetcode322

```shell
示例 1：
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
示例 2：
输入：coins = [2], amount = 3
输出：-1
示例 3：
输入：coins = [1], amount = 0
输出：0

```

代码：

```c++
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        // coins = [1, 2, 5], amount = 11
        vector<long long> f(amount+1);
        for(int i=1; i<=amount; i++){
            f[i] = INT_MAX;
            for(int c:coins){
                if(i>=c) f[i] = min(f[i-c]+1, f[i]);
            }
        }
        return f[amount]==INT_MAX?-1:f[amount];
    }
};
// 两层循环的遍历顺序和上面不同
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<long long> f(amount+1, INT_MAX-1);
        f[0] = 0;
        for (int c:coins){
            for(int i=1; i<=amount; i++){
                if (i>=c) f[i] = min(f[i-c]+1, f[i]);
            }
        }
        return f[amount]==INT_MAX-1?-1:f[amount];
    }
};
// 优化i>=c
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<long long> f(amount+1, INT_MAX-1);
        f[0] = 0;
        for (int c:coins){
            for(int i=c; i<=amount; i++){
                f[i] = min(f[i-c]+1, f[i]);
            }
        }
        return f[amount]==INT_MAX-1?-1:f[amount];
    }
};
```



## 2.5 最大子序和_leetcode53

```shell
给定一个整数数组nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

示例 2：
输入：nums = [1]
输出：1
```

代码：

```c++
注意2点:1、需要修改原数组nums[i]为当前最大值；2、last记录的是上一次的最大值；
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        // [-2,1,-3,4,-1,2,1,-5,4]
        int n = nums.size();
        int res = INT_MIN, last = 0;
        for(int i=0; i<n; i++){
            nums[i] += max(0, last); // nums[i]为当前最大值
            res = max(res, nums[i]);
            last = nums[i];
        }
        return res;
    }
};
```

## 2.6 不同路径_leetcode62_63

```shell
一个机器人位于一个m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
问总共有多少条不同的路径？

示例 2：
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下

示例 3：
输入：m = 7, n = 3
输出：28

```

f[i,j]表示从start到[i,j]的路径数；f[i, j]=f[i-1,j]+f[i,j-1];

代码:

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> f(m, vector<int>(n));
        for(int i=0; i<m; i++) f[i][0] = 1;
        for(int i=0; i<n; i++) f[0][i] = 1;
        for(int i=1; i<m; i++){
            for(int j=1; j<n; j++){
                f[i][j] = f[i-1][j] + f[i][j-1];
            }
        }
        return f[m-1][n-1];
    }
};
// 空间优化从O(m*n)到O(n)==>第i行只是依赖于i-1行
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> f(n);
        for(int i=0; i<n; i++) f[i] = 1;
        for(int i=1; i<m; i++){
            f[0] = 1;
            for(int j=1; j<n; j++){
                f[j] += f[j-1];
            }
        }
        return f[n-1];
    }
};
// 有障碍物63
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size(), n = obstacleGrid[0].size();
        vector<vector<int>> f(m, vector<int>(n));
        for(int i=0; i<m; i++){
            if(obstacleGrid[i][0]==0) f[i][0] = 1;
            else break;
        }
        for(int i=0; i<n; i++){
            if(obstacleGrid[0][i]==0) f[0][i] = 1;
            else break;
        }
        for(int i=1; i<m; i++){
            for(int j=1; j<n; j++){
                if(obstacleGrid[i][j]==0) f[i][j] = f[i-1][j] + f[i][j-1];
            }
        }
        return f[m-1][n-1];      
    }
};
```

## 2.7 最小路径和_leetcode64

给定一个包含非负整数的$m*n$网格grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

```shell
示例 1：
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。

示例 2：
输入：grid = [[1,2,3],[4,5,6]]
输出：12

```

状态定义，状态转换方程，初始化；

代码：

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        // [[1,3,1],[1,5,1],[4,2,1]]
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> f(m, vector<int>(n));// 状态表示f[i][j]表示从[0,0]到[i,j]的最小值
        f[0][0] = grid[0][0];
        // 初始化
        for(int i=1 ;i<m; i++) f[i][0] = grid[i][0] + f[i-1][0];
        for(int i=1 ;i<n; i++) f[0][i] = grid[0][i] + f[0][i-1];
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                f[i][j] = min(f[i][j-1], f[i-1][j]) + grid[i][j]; // 状态计算
            }
        }
        return f[m-1][n-1];
    }
};
```

## 2.8 解码方法_leetcode91

```shell
示例 1：
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。

示例 2：
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
```

状态定义与状态转换方程；

代码:

```c++
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        vector<int> f(n+1);
        f[0] = 1;
        for(int i=1; i<=n; i++){
            if (s[i-1] != '0') f[i] += f[i-1]; // f[i]=0,因此，先将f[i-1]赋值给[i]
            if (i >= 2) {
                int t = 10*(s[i-2]-'0')+s[i-1]-'0';
                if(t>=10 && t<=26) f[i] += f[i-2];
            }
        }
        return f[n];
    }
};
```

## 2.9 打家劫舍_leetcode198

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

```shell
示例 1：
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。

示例 2：
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

代码：

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        vector<int> f(n+1), g(n+1); // [2,7,9,3,1]
        // f[i]表示不选择当前房间nums[i]的最大值；g[i]表示选择当前房间nums[i]的最大值
        for(int i=1; i<=n; i++) {
            f[i] = max(g[i-1], f[i-1]);
            g[i] = f[i-1] + nums[i-1];
        }
        return max(f[n], g[n]);
    }
};
```

# 三、string

## 3.1 单词拆分_leetcode139

```shell
给定一个非空字符串s和一个包含非空单词的列表wordDict，判定s是否可以被空格拆分为一个或多个在字典中出现的单词。
说明：
拆分时可以重复使用字典中的单词。你可以假设字典中没有重复的单词。
示例 1：
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
示例 2：
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
示例 3：
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

https://blog.csdn.net/liugg2016/article/details/82119498?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.baidujs&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.baidujs;

思路:

```shell
# 1、动态规划
定义dp[i]表示字符串s前i个字符组成的字符串s[0..i-1]是否能被空格拆分成若干个字典中出现的单词。从前往后计算考虑转移方程，每次转移的时候我们需要枚举包含位置i−1的最后一个单词，看它是否出现在字典中以及除去这部分的字符串是否合法即可。公式化来说，我们需要枚举 s[0..i-1]中的分割点j，看s[0..j-1]组成的字符串s1(默认j=0时s_1为空串)和s[j..i-1]组成的字符串s_2是否都合法。如果两个字符串均合法，那么按照定义s_1和s_2拼接成的字符串也同样合法。由于计算到dp[i]时我们已经计算出了dp[0..i−1]的值，因此字符串s_1是否合法可以直接由dp[j]得知，剩下的我们只需要看s_2是否合法即可，因此我们可以得出如下转移方程：dp[i]=dp[j] && check(s[j..i−1]);其中 check(s[j..i−1])表示子串s[j..i−1]是否出现在字典中。
对于检查一个字符串是否出现在给定的字符串列表里一般可以考虑哈希表来快速判断，同时也可以做一些简单的剪枝，枚举分割点的时候倒着枚举，如果分割点j到i的长度已经大于字典列表里最长的单词的长度，那么就结束枚举，但是需要注意的是下面的代码给出的是不带剪枝的写法。
对于边界条件，我们定义dp[0]=true表示空串且合法。
# 2. 记忆化回溯
使用记忆化函数，保存出现过的backtrack(s)，避免重复计算。
定义回溯函数backtrack(s)
若s长度为0，则返回True，表示已经使用wordDict中的单词分割完。
初试化当前字符串是否可以被分割res=False；
遍历结束索引i，遍历区间[1,n+1)；
若 s[0,...,i-1]在wordDictwordDict中:res=backtrack(s[i,...,n-1]) or res。解释：保存遍历结束索引中，可以使字符串切割完成的情况。
返回res
返回 backtrack(s)
```

代码：

```c++
// 1、动态规划
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> wordSets = unordered_set<string>();
        for (auto word: wordDict) {
            wordSets.insert(word);
        }
        int n = s.size();
        vector<bool> f = vector<bool>(n+1);
        f[0] = true;
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j < i; ++j) {
                //f[j]表示s[0,...,j-1]能被切分， wordSets.find(s.substr(j, i - j))表示剩下的能否被切分
                if (f[j] && wordSets.find(s.substr(j, i - j)) != wordSets.end()){
                    f[i] = true;
                    break;
                }
            }
        }
        return f[n];
    }
};
```

时间复杂度:$O(n^2)$;空间复杂度$O(n)$;

## 3.2 单词拆分_leetcode140

```shell
给定一个非空字符串s和一个包含非空单词列表的字典wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。
说明：
分隔时可以重复使用字典中的单词。你可以假设字典中没有重复的单词。
示例 1：
输入:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
输出:
["cats and dog", "cat sand dog"]
示例 2：
输入:
s = "pineapplepenapple"
wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
输出:
[
  "pine apple pen apple","pineapple pen apple","pine applepen apple"]
解释: 注意你可以重复使用字典中的单词。

```

代码：

```c++
class Solution {
public:
	vector<string> wordBreak(string s, vector<string>& wordDict)
	{
		if (!wordBreak_139(s, wordDict)) return {};

		size_t validEnd = 0;
		vector<vector<string>> dp(s.size() + 1, vector<string>());

		for (size_t i = 0; i < s.size(); i++)
		{
			if (i == validEnd + 1) return {};
			if (i != 0 && dp[i].empty()) continue;
			for (auto& word : wordDict)
			{
				size_t newEnd = i + word.size();
				if (newEnd > s.size()) continue;
				if (memcmp(&s[i], &word[0], word.size()) != 0) continue;
				validEnd = max(validEnd, newEnd);
				if (i == 0)
				{
					dp[newEnd].push_back(word);
					continue;
				}
				for (auto& d : dp[i])
				{
					dp[newEnd].push_back(d + " " + word);
				}
			}
		}

		return dp.back();
	}
  
  // "catsanddog"；["cats", "dog", "sand", "and", "cat"]
	bool wordBreak_139(string& s, vector<string>& wordDict)
	{
		size_t validEnd = 0;
		vector<bool> dp(s.size() + 1, false);
		dp[0] = true;
		for (size_t i = 0; i < s.size(); i++)
		{
			if (i == validEnd + 1) return false;
			if (!dp[i]) continue;
			for (auto& word : wordDict) // 遍历单词集合
			{
				size_t newEnd = i + word.size();
				if (newEnd > s.size()) continue;
				if (memcmp(&s[i], &word[0], word.size()) == 0)
				{
					dp[newEnd] = true;
					validEnd = max(validEnd, newEnd);
				}
			}
		}
		return dp.back();
	}
};
```







