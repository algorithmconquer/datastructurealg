# 一、原理

状态定义，状态转移方程；

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

主要注意替换replace;

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
                f[i][j] = min(f[i-1][j], f[i][j-1])+1;
                // 替换
                f[i][j] = min(f[i-1][j-1]+(word1[i-1]!=word2[j-1]), f[i][j]);
            }
        }
        return f[m][n];
    }
};
```

