# 746 ------leetcode---------使用最小花费爬楼梯

给你一个整数数组 `cost` ，其中 `cost[i]` 是从楼梯第 `i` 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。

你可以选择从下标为 `0` 或下标为 `1` 的台阶开始爬楼梯。

请你计算并返回达到楼梯顶部的最低花费。

```python
def minCostClimbingStairs(cost):
    n = len(cost)
    if n == 0:
        return 0
    if n == 1:
        return cost[0]
    
    # Initialize dp array
    dp = [0] * n
    dp[0] = cost[0]
    dp[1] = cost[1]
    
    # Fill dp array
    for i in range(2, n):
        dp[i] = min(dp[i-1] + cost[i], dp[i-2] + cost[i])
        
    # The result is the minimum cost to reach the top
    return min(dp[n-1], dp[n-2])

# Example usage:
cost = [10, 15, 20]
print(minCostClimbingStairs(cost))  # Output: 15

```



 这道题可以使用动态规划（Dynamic Programming，简称DP）来解决。我们定义一个数组 `dp`，其中 `dp[i]` 表示到达第i个台阶所需的最小花费。根据题目要求，我们可以从第0个台阶或第1个台阶开始爬楼梯，因此初始化如下：

- `dp[0] = cost[0]`：从第0个台阶开始的最小花费就是 `cost[0]`。
- `dp[1] = cost[1]`：从第1个台阶开始的最小花费就是 `cost[1]`。

对于其他的台阶i（i >= 2），有两种选择：

1. 从第i-1个台阶跨一步到达第i个台阶，此时的花费为 `dp[i-1] + cost[i]`。
2. 从第i-2个台阶跨两步到达第i个台阶，此时的花费为 `dp[i-2] + cost[i]`。

因此，状态转移方程为： dp[i]=min⁡(dp[i−1]+cost[i],dp[i−2]+cost[i])dp[i] = \min(dp[i-1] + cost[i], dp[i-2] + cost[i])dp[i]=min(dp[i−1]+cost[i],dp[i−2]+cost[i])

最终，我们要求的结果是 `dp[n-1]`，其中n为台阶的总数，即最后一个台阶的最小花费。







在这段代码中，我们首先检查数组长度，然后初始化 `dp` 数组并填充它，最后返回 `dp[n-1]` 和 `dp[n-2]` 中的较小值作为最终答案。





从第i-1个台阶跨一步到达第i个台阶，此时的花费为 dp[i-1] + cost[i]。 为什么呢？



ChatGPT

对于动态规划中台阶问题的状态转移方程：

dp[i]=min⁡(dp[i−1]+cost[i],dp[i−2]+cost[i])dp[i] = \min(dp[i-1] + cost[i], dp[i-2] + cost[i])dp[i]=min(dp[i−1]+cost[i],dp[i−2]+cost[i])

让我们详细解释为什么从第i-1个台阶跨一步到达第i个台阶时，其花费是 dp[i−1]+cost[i]dp[i-1] + cost[i]dp[i−1]+cost[i]。

假设我们已经计算出了到达第i-1个台阶的最小花费 dp[i−1]dp[i-1]dp[i−1]，现在我们要到达第i个台阶。有两种选择：

1. **从第i-1个台阶跨一步到达第i个台阶：** 这意味着我们从第i-1个台阶跨了一步到达第i个台阶，因此花费是当前台阶i的费用 cost[i]cost[i]cost[i]，再加上到达第i-1个台阶的最小花费 dp[i−1]dp[i-1]dp[i−1]，所以总花费是 dp[i−1]+cost[i]dp[i-1] + cost[i]dp[i−1]+cost[i]。
2. **从第i-2个台阶跨两步到达第i个台阶：** 这意味着我们从第i-2个台阶跨了两步到达第i个台阶，因此花费是当前台阶i的费用 cost[i]cost[i]cost[i]，再加上到达第i-2个台阶的最小花费 dp[i−2]dp[i-2]dp[i−2]，所以总花费是 dp[i−2]+cost[i]dp[i-2] + cost[i]dp[i−2]+cost[i]。

我们的目标是找到到达最后一个台阶（顶部）的最小花费，因此我们取上述两种选择中的最小值作为 dp[i]dp[i]dp[i] 的值。

这种设计确保了每个状态 dp[i]dp[i]dp[i] 都代表了从起点（第0个或第1个台阶）到达第i个台阶的最小花费，利用了子问题的最优解性质，从而构建出整体的最优解。





