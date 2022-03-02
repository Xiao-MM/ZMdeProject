package algorithm;

import java.util.*;

public class Algorithms {
    /**
     * 获取整数的位数
     * @param x
     * @return
     */
    public static int getNumLength(int x){
        int n;
        if (x < 0) x = -x;
        for (n = 0; x > 0; n++){
            x = x/10;
        }
        return n;
    }

    /**
     * 7. 整数反转
     * digit = x%10
     * result = result*10 + digit;
     * @param x 带翻转数
     * @return
     */
    public static int reverse(int x){
        int digit = 0;// 取模得到的尾数
        int result = 0;// 需要累乘的结果
        while (x != 0){
            //因为x本身会被int限制，当x为正数并且位数和Integer.MAX_VALUE的位数相等时首位最大只能为2，
            // 所以逆转后不会出现res = Integer.MAX_VALUE / 10 && tmp > 2的情况，
            // 自然也不需要判断res==214748364 && tmp>7了，反之负数情况也一样
            if (result < Integer.MIN_VALUE/10 || result > Integer.MAX_VALUE/10){
                return 0;
            }
            digit = x%10;
            x/=10;
            result = result*10 + digit;
        }
        return result;
    }

    /**
     * 1. 两数之和
     * @param nums 列表
     * @param target 目标数字
     * @return
     */
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])){
                return new int[]{map.get(target - nums[i]),i};
            }
            map.put(nums[i],i);
        }
        return new int[]{0};
    }

    /**
     * 300. 最长递增子序列
     * @param nums
     * @return
     */
    public static int lengthOfLIS(int[] nums){
        int[] d = new int[nums.length];
        int max = d[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            d[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && d[i] <= d[j]){
                    d[i] = d[j] + 1;
                }
            }
            if (max < d[i]){
                max = d[i];
            }
        }
        return max;
    }

    /**
     * 300. 最长递增子序列
     * 官方答案
     * @param nums
     * @return
     */
    public static int officialLengthOfLIS(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int maxans = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            maxans = Math.max(maxans, dp[i]);
        }
        return maxans;
    }

    /**
     * 动态规划求解斐波那契
     * @param n
     * @return
     */
    public static int fib(int n) {
        if (n < 2){
            return n;
        }
        int p = 0, q = 1, k = 0;
        for (int i = 2; i <= n; i++) {
            p = q;
            q = k;
            k = p + q;
        }
        return k;
    }

    /**
     * 矩阵法求斐波那契
     * @param n
     * @return
     */
    public static int fib_matrix(int n) {
       int[][] base = new int[][]{{1,1},{1,0}};
       int[][] baseN = matrixFastMulti(base,n);
       return baseN[1][0];// 该元素即为所求
    }

    /**
     * 快速幂算法
     * 7^10 = 7^5*7^5
     * 适用于数和矩阵，满足结合律的类型
     * 其时间复杂度为 o(log n)
     * @param data 待乘数据
     * @param n 幂
     * @return
     */
    public static int fastMulti(int data, int n){
        int result = 1;
        while (n > 0){
            if ((n & 1) == 1){ //判断二进制最后一位是否为1
                result *= data;
            }
            data *= data;
            n >>= 1;
        }
        return result;
    }

    /**
     * 50. Pow(x, n)
     * @param x
     * @param n
     * @return
     */
    public static double myPow(double x, int n) {
        if (x == 0.0){
            return 0;
        }
        long b = n;// -2^32 转 2^32会发生越界
        // 将负数幂转成正数幂
        if (n < 0){
            x = 1/x; // 将
            b = -b;
        }
        double result = 1;
        while (b > 0){
            if ((b & 1) == 1){
                result *= x;
            }
            x *= x;
            b >>= 1;
        }
        return result;
    }

    /**
     * 矩阵快速幂（二阶）
     * @param data
     * @param n
     * @return
     */
    public static int[][] matrixFastMulti(int[][] data, int n){
        int[][] result = new int[][]{{1,0},{0,1}};// 初始化单位矩阵
        while (n > 0){
            if ((n & 1) == 1){
                result = matrixMulti(result,data);
            }
            data = matrixMulti(data,data);
            n >>= 1;
        }
        return result;
    }
    /**
     * 二阶矩阵乘法
     * @param a
     * @param b
     * @return
     */
    public static int[][] matrixMulti(int[][] a, int[][] b){
        int[][] result = new int[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
            }
        }
        return result;
    }

    /**
     * 9. 回文数
     * 1. 2221/1=2221>10,div=10
     * 2. 2221/10=222>10,div=10*10=100
     * 3. 2221/100=22>10,div=100*10=1000
     * 4. 2221/1000=2<10,break div = 1000
     * @param x
     * @return
     */
    public static boolean isPalindrome(int x) {
        if (x < 0) return false;
        int div = 1;// 初始化除数
        while (x/div >= 10) div *= 10;// 获取最大除数
        while (x > 0){
            int left = x/div;// 获取最高位的值 2221->2
            int right = x%10;// 获取最低位的值 2221->1
            if (left != right) return false;// 2 != 1
            x = (x % div)/10;// 1221 -> 22
            div /= 100;// 1000->10
        }
        return true;
    }

    /**
     * 5. 最长回文子串 暴力解法, 严重超时o(n^3)
     * @param s
     * @return
     */
    public static String longestPalindromeViolate(String s) {
        if (s.length() <= 1){
            return s;
        }
        int maxLength = 0;
        String maxSubStr = "";
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                String substring = s.substring(j, i);
                if (isPalindrome(substring) && substring.length() > maxLength){
                    maxLength = substring.length();
                    maxSubStr = substring;
                }
            }
        }
        return maxSubStr;
    }

    /**
     * 5. 最长回文子串
     * 动态规划思想 p(i,j) 表示下标从i到j的字符串时是否为回文串
     * 状态转移方程 p(i,j) = p(i+1,j-1)&&(s[i]==s[j])
     * @param s j-1 - (i+1) + 1 = j-i-1
     * @return
     */
    public static String longestPalindrome(String s) {
        // 长度小于2直接就是回文串
        if (s == null || s.length() < 2){
            return s;
        }
        char[] chars = s.toCharArray();
        boolean[][] dp = new boolean[s.length()][s.length()];
        // 边界值，单个字符串是回文串
        for (int i = 0; i < s.length(); i++) {
            dp[i][i] = true;
        }
        // 待截取子串起始位置
        int begin = 0;
        // 待截取子串的最大长度,需初始化为1
        int maxLen = 1;
        // 根据字符串长度枚举判断子串是否为回文串 分别是从 2 -> s.length的子串
        for (int len = 2; len <= chars.length; len++) {
            // 枚举左边界
            for (int l = 0; l < chars.length; l++) {
                // 根据左边界和串长确定右边界
                int r = len + l - 1;
                // 右边界越界直接跳出循环
                if (r > chars.length - 1){
                    break;
                }
                // 如果l和r处的字符不相等则该范围串非回文串
                if (chars[l] != chars[r]){
                    dp[l][r] = false;
                }else {
                    // 如果串长为2和3且char[l]==char[r]该串为回文串
                    if (len <= 3){// len <= 3
                        dp[l][r] = true;
                    }else {
                        // len > 3 时由 dp[l+1][r-1]值决定
                        dp[l][r] = dp[l+1][r-1];
                    }
                }
                // 如果l->r为回文串更新最大串长
                if (dp[l][r] && len > maxLen){
                    maxLen = len;
                    begin = l;
                }
            }
        }
        return s.substring(begin, begin+maxLen);
    }

    /**
     * 3.无重复字符的最长子串
     * @param s
     * @return
     */
    public static int lengthOfLongestSubstring(String s) {
        if (s.length() == 0){
            return 0;
        }
        HashMap<Character,Integer> map = new HashMap<>();
        int maxLen = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++) {
            if (map.containsKey(s.charAt(right))){
                left = Math.max(left, map.get(s.charAt(right)));// 如果map中存在已知元素，更新左边界下标
            }
            map.put(s.charAt(right), right + 1);// map中存放滑动窗口left下一个移动的位置
            maxLen = Math.max(right - left + 1, maxLen);
        }

        return maxLen;
    }

    /**
     * 15. 三数之和
     * @param nums
     * @return
     */
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null || nums.length < 3){//小于3的数组直接返空
            return result;
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {// 固定i在 [i,nums.length-1]中匹配剩下两个数
//            if(nums[i] > 0) break; // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
            if (nums[i] > 0) return result;
            int l = i + 1, r = nums.length - 1;
            if (i > 0 && nums[i] == nums[i-1]) continue; // 去重
            while (l < r){
                int sum = nums[i] + nums[l] + nums[r];
                if (sum > 0){ //和大于0，正确答案一定在右边界左边
                    r--;
                }else if (sum < 0){ //和小于0，正确答案一定在左边界右边
                    l++;
                }else {
                    result.add(Arrays.asList(nums[i],nums[l],nums[r]));
                    while (l < r && nums[l] == nums[l+1]) l++; // 去重
                    while (l < r && nums[r] == nums[r-1]) r--; // 去重
                    l++;
                    r--;
                }
            }
        }
        return result;
    }

//    /**
//     * 120. 三角形最小路径和
//     * 动态规划解法 dp[i,j] = min(dp[i+1,j],dp[i+1,j+1] + triangle[i,j])
//     * @param triangle
//     * @return
//     */
//    public static int minimumTotal(List<List<Integer>> triangle) {
//        int[][] dp = new int[triangle.size()][triangle.size()];
//        for (int i = triangle.size() - 1; i >= 0 ; i--) {
//            for (int j = 0; j <= i; j++) {
//                if (i == triangle.size() - 1){
//                    dp[i][j] = triangle.get(i).get(j);
//                }else {
//                    dp[i][j] = Math.min(dp[i+1][j], dp[i+1][j+1]) + triangle.get(i).get(j);
//                }
//            }
//        }
//        return dp[0][0];
//    }

    /**
     * 120. 三角形最小路径和
     * 动态规划解法 dp[i,j] = min(dp[i+1,j],dp[i+1,j+1] + triangle[i,j])
     * 优化空间
     * @param triangle
     * @return
     */
    public static int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        // 最底层n个元素，定义n + 1可以防止初始 dp[j + 1] 越界，滚动数组一开始只是最底层用n，一次循环结束后上一层用n-1，以此循环向上
        int[] dp = new int[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            // 每层有下标 0->i个元素
            for (int j = 0; j <= i; j++) {
                dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }

//    /**
//     * 62. 不同路径
//     * @param m
//     * @param n
//     * @return
//     */
//    public static int uniquePaths(int m, int n) {
//        int[][] dp = new int[m][n];
//        // 初始化第一列
//        for (int i = 0; i < m; i++) {
//            dp[i][0] = 1;
//        }
//        // 初始化第一行
//        for (int i = 0; i < n; i++) {
//            dp[0][i] = 1;
//        }
//        for (int i = 1; i < m; i++) {
//            for (int j = 1; j < n; j++) {
//                dp[i][j] = dp[i-1][j] + dp[i][j-1];
//            }
//        }
//        return dp[m-1][n-1];
//    }
    /**
     * 62. 不同路径
     * 空间优化
     * @param m
     * @param n
     * @return
     */
    public static int uniquePaths(int m, int n) {
        int[] dp = new int[n];// 优化成一维数组，因为每次计算上一行都不需要再次使用
        for (int i = 0; i < n; i++) {
            dp[i] = 1;// 初始化1
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j-1];
            }
        }
        return dp[n-1];
    }
//        int eatAppleNum = 0; // 吃掉苹果的数量
//        int canEetAppleNum = 0; // 可以吃的苹果的数量
//        int minDays = 0; // 还有几天过期
//        int invalidDay = 0; // 最近过期日
//        // 给苹果绑定过期时间map(k,v) k=第几天，v=数量
//        Map<Integer,Integer> map = new HashMap<>();// 记录第几天的苹果有几天的保质期
//        for (int i = 0; i < apples.length; i++) {
//            map.put(i, apples[i]);// 将第i天的苹果和其数量放入Map
//            // 每过一天检查前几天的苹果是否过期
//            for (int j = 0; j < i; j++) {
//                if (days[i] >= i - j){
//                    map.put(j,0);// 置第i天的苹果数量为0
//                }
//                int leave = days[i] - i + j;
//                minDays = Math.min(leave,minDays);
//                invalidDay = j;
//                canEetAppleNum += map.get(j);
//                //这时要吃掉最快过期的苹果
//            }
//            // 如果有可以吃的苹果
//            if (canEetAppleNum > 0){
//                canEetAppleNum --; // 吃掉一个苹果
//                map.put(invalidDay, map.get(invalidDay)-1);// 置那天的苹果数量少一个
//            }
//        }
    /**
     * 1705. 吃苹果的最大数目
     * 贪心 + 小顶堆
     * @param apples
     * @param days
     * @return
     */
    public static int eatenApples(int[] apples, int[] days) {
        int length = apples.length;
        int eatAppleNum = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));// 定义小根堆,元素比较策略根据数组第一个元素
        // 小顶堆存放[过期时间，该时间存放的苹果]
        // 数组长度期间的吃法,过期日期 = days[day] + day
        int day;
        for (day = 0; day < length; day++) {
            // 如果当前日期有苹果产生，将其连同过期日期和数量加入小顶堆
            if (apples[day] > 0){
                pq.offer(new int[]{days[day] + day, apples[day]});
            }
            // 如果队列有苹果，并且当前苹果过期了则移除该元素，否则该天苹果数量-1，如果减到0则移除
            while (pq.size() > 0){
                // 贪心策略，每次都吃最快过期的苹果
                int[] top = pq.peek();
                // 过期了移除
                if (top[0] <= day){
                    pq.remove();
                }else {
                    top[1]--;// 苹果数量-1
                    eatAppleNum ++;
                    if (top[1] == 0){ // 苹果吃完了移除该天的苹果
                        pq.remove();
                    }
                    break;
                }
            }
        }
        // 数组长度结束期间的吃法，当堆不空时，选堆顶元素，可以吃的数量 = min(过期日期-当前天数，苹果数)
        while (pq.size() > 0){
            int[] top = pq.peek();
            // 过期了移除
            if (top[0] <= day){
                pq.remove();
            }
            top = pq.peek(); //更新top
            if (top == null) break;
            int min = Math.min(top[0] - day, top[1]);// 计算还可以吃的苹果数
            day += min;// 日期推迟到苹果吃完
            eatAppleNum += min;
            pq.remove();
        }
        return eatAppleNum;
    }
//        int length = prices.length;
//        int[] p = new int[length+1];
//        for (int i = 0; i < length; i++) {
//            p[i] = prices[i];
//        }
//    /**
//     * 121. 买卖股票的最佳时机
//     * 单调栈实现 妈的整的贼麻烦，思想是好的，效率真吉尔差
//     * 执行用时：40 ms, 在所有 Java 提交中击败了7.44%的用户
//     * 内存消耗：51.1 MB, 在所有 Java 提交中击败了85.54%的用户
//     * @param prices
//     * @return
//     */
//    public static int maxProfit(int[] prices) {
//        prices = Arrays.copyOf(prices, prices.length + 1);//原数组扩容一个元素
////        prices[prices.length-1] = 0;// 哨兵
//        int profit = 0;
//        ArrayDeque<Integer> stack = new ArrayDeque<>();
////        Stack<Integer> stack = new Stack<>();//维护单调栈
//        for (int i = 0; i < prices.length; i++) {
//            if (!stack.isEmpty() && prices[i] <= prices[i - 1]) {
//                while (!stack.isEmpty() && stack.peek() > prices[i]) {
//                    Integer top = stack.pop();// 出栈
//                    if (stack.isEmpty()) continue;
////                    profit = Math.max(profit, top - stack.firstElement());// 出栈元素-栈底元素 确认最大收益
//                    profit = Math.max(profit, top - stack.peekLast());// 出栈元素-栈底元素 确认最大收益
//                }
//            }
//            stack.push(prices[i]);// 当后者大于前者入栈
//        }
//        return profit;
//    }

    /**
     *  121. 买卖股票的最佳时机
     * @param prices
     * @return
     */
    public static int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int price : prices) {
            if (price < minPrice) {
                minPrice = price;
            } else if (price - minPrice > maxProfit) {
                maxProfit = price - minPrice;
            }
        }
        return maxProfit;
    }

    /**
     * 122. 买卖股票的最佳时机 II
     * @param prices
     * @return
     */
    public static int maxProfit2(int[] prices) {
        // profit 为收益， tmp 为 相隔两天之间的收益
        int profit = 0;
        // 购买策略为只要任意两天呈上涨趋势就买入和卖出，否则就不买，（跳过）
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]){
                profit += prices[i] - prices[i - 1];
            }
        }
        return profit;
    }

    /**
     * 240. 搜索二维矩阵 II
     * 剑指 Offer 04. 二维数组中的查找
     * 时间 o(m+n) 空间 o(1)
     * @param matrix
     * @param target
     * @return
     */
    public static boolean findNumberIn2DArray(int[][] matrix, int target) {
        int i = matrix.length - 1, j = 0;
        while (i >= 0 && j < matrix[0].length){
            if (matrix[i][j] > target) i--;
            else if (matrix[i][j] < target) j++;
            else return true;
        }
        return false;
    }

    /**
     * 240. 搜索二维矩阵 II
     * o(m*log n)
     * @param matrix
     * @param target
     * @return
     */
    public static boolean searchMatrix2(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; i++) {
            int l = 0, r = n - 1;
            int mid;
            while (l <= r){ // 注意二分的条件是 l <= r
                mid = (l + r) >> 1;
                if (matrix[i][mid] > target){
                    r = mid - 1;
                }else if (matrix[i][mid] < target){
                    l = mid + 1;
                }else {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 剑指 Offer 60. n个骰子的点数
     * f(n,x) = ∑(1->6) f(n-1,x-i)*(1/6)
     * dp[n,x] = dp[n,x] + dp[n-1,x-i]*(1/6)
     * 时间 o(n^2),空间 o(n)
     * @param n 骰子的个数
     * @return
     */
    public static double[] dicesProbability(int n) {
        double[] dp = new double[6];// 表示当前筛子个数下所有数量和的概率，数组下标 (0, n - 1) 表示（n, 2n）
        Arrays.fill(dp, 1.0/6.0);
        for (int i = 2; i <= n; i++) { //从2一次地推到n
            double[] temp = new double[5 * i + 1];// 所有的骰子的和种类数，n个筛子有6n - n + 1 种筛子和
            for (int j = 0; j < dp.length; j++) {// 依次遍历dp，用 筛子个数 i 的每个种类和 j 去递推 筛子i+1的种类和为 j+k 的概率值dp[j+k]
                for (int k = 0; k < 6; k++) {// 新筛子的数只可能是1-6，循环6次即可
                    temp[j + k] += dp[j]/6.0; // 对dp[j]/6求和
                }
            }
            dp = temp;// 用新的结果替换上一层的结果
        }
        return dp;// 返回最新一层数据
    }

//    /**
//     * 6. Z 字形变换
//     * 该写法代码显得过于零散
//     * @param s 给定字符串 s
//     * @param numRows 根据给定的行数 numRows
//     * @return 输出需要从左往右逐行读取，产生出一个新的字符串
//     */
//    public static String convert(String s, int numRows) {
//        if (numRows < 2){
//            return s;
//        }
//        List<StringBuilder> temps = new ArrayList<>();
//        for (int i = 0; i < numRows; i++) {
//            temps.add(new StringBuilder());
//        }
//        StringBuilder result = new StringBuilder();
//        boolean flag = true;// 当为true，正向往数组写数据，false反向向数组写数据
//        int count = 0;
//        char[] chars = s.toCharArray();
//        for (char aChar : chars) {
//            temps.get(count).append(aChar);
//            if (flag) {
//                count++;
//            } else {
//                count--;
//            }
//            if (count == numRows - 1) {
//                flag = false;
//            }
//            if (count == 0) {
//                flag = true;
//            }
//        }
//        for (StringBuilder temp : temps) {
//            result.append(temp);
//        }
//        return result.toString();
//    }
    /**
     * 6. Z 字形变换
     * @param s 给定字符串 s
     * @param numRows 根据给定的行数 numRows
     * @return 输出需要从左往右逐行读取，产生出一个新的字符串
     */
    public static String convert(String s, int numRows) {
        if (numRows < 2){
            return s;
        }
        List<StringBuilder> temps = new ArrayList<>();// 注意这里不能使用数组，StringBuilder[i]会将所有的数组元素视为同一个对象
        for (int i = 0; i < numRows; i++) {
            temps.add(new StringBuilder());// 初始化每个数组对象
        }
        StringBuilder result = new StringBuilder();
        int count = 0;
        int flag = 1;// 用数字表示状态字可以用来参与运算
        for (char c : s.toCharArray()) {
            temps.get(count).append(c);// 写入第count行
            count += flag;
            if (count == 0 || count == numRows - 1){// 满足该条件就要掉头反向写数据了
                flag = - flag;
            }
        }
        for (StringBuilder temp : temps) {
            result.append(temp);
        }
        return result.toString();
    }

    /**
     * 8. 字符串转换整数 (atoi)
     * @param s
     * @return
     */
    public static int myAtoi(String s) {
        if (s == null || s.length() == 0) return 0;
        List<Character> baseNumber = Arrays.asList('0','1','2','3','4','5','6','7','8','9');
        char[] chars = s.trim().toCharArray();
        boolean isFirstSearch = false;
        long result = 0L;
        int sign = 1;// 初始符号位，默认为+
        for (char c : chars) {
            // 1. 取出前导空格，没有空格则下一步
//            if (c == ' '){
//                continue;
//            }
            // 2. 取第一个符号，如果+，-则+，-，否则按+处理
            if (!isFirstSearch){// 只做一个扫描
                isFirstSearch = true;
                // 整数不变继续扫描下一个元素
                if (c == '-'){// 只要没有出现'-'就按整数处理
                    sign = - sign;
                    continue;// 确认完符号继续扫描下一个元素
                }
                if (c == '+') continue;// 确认完符号继续扫描下一个元素

            }
            // 3. 依次向后取数字，
            if (!baseNumber.contains(c)){
                // 如果取的不是数字则返回已经转化过的数字
                return (int) result;
            }
            // 4. 将取出来的数字依次累乘10相加直到遇到下一个字符非数字，判断前面的数字是否越界，如未越界则返回
            // 5. 如果发生越界则需要返回Integer.MAX_VALUE或Integer.MIN_VALUE，注意越界需要提前判断
            result = result * 10 + sign * Integer.parseInt(String.valueOf(c));
            if (result > Integer.MAX_VALUE) return Integer.MAX_VALUE;
            if (result < Integer.MIN_VALUE) return Integer.MIN_VALUE;
        }
        return (int) result;
    }

//    /**
//     * 1765. 地图中的最高点
//     * 多源广度优先遍历数组
//     * @param isWater 如果 isWater[i][j] == 0 ，格子 (i, j) 是一个 陆地 格子。如果 isWater[i][j] == 1 ，格子 (i, j) 是一个 水域 格子。
//     * @return
//     */
//    public static int[][] highestPeak(int[][] isWater) {
//        // 方向数组，分别表示向左，向右，向上，向下移动
//         int[][] directions = new int[][]{{-1, 0},{1, 0},{0, -1},{0, 1}};
//        int m = isWater.length, n = isWater[0].length;
//        // 结果集
//        int[][] result = new int[m][n];
//        // 初始结果为 -1
//        for (int i = 0; i < m; i++) {
//            Arrays.fill(result[i], -1);
//        }
//        // 初始化队列，队列存储矩阵的位置坐标
//        Queue<int[]> queue = new ArrayDeque<>();
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < n; j++) {
//                if (isWater[i][j] == 1){
//                    result[i][j] = 0;// 初始化海洋
//                    queue.offer(new int[]{i, j});// 位置信息入队
//                }
//            }
//        }
//        // 开始广度遍历（层次）
//        while (!queue.isEmpty()){
//            int[] p = queue.poll();// 出队
//            // 对出队的每个元素向周围扩散一圈
//            for (int[] dir : directions) {
//                int x = p[0] + dir[0], y = p[1] + dir[1];
//                // 范围判断，未被访问
//                if (x >= 0 && x < m && y >= 0 && y < n && result[x][y] == -1){
//                    result[x][y] = result[p[0]][p[1]] + 1;
//                    queue.offer(new int[]{x, y});// 新元素入队
//                }
//            }
//        }
//        return result;
//    }

    /**
     * 1765. 地图中的最高点
     * 牛逼
     * 题目可以转化为到0（海域）的最近距离矩阵
     * @param isWater
     * @return
     */
    public static int[][] highestPeak(int[][] isWater) {
        int m = isWater.length;
        int n = isWater[0].length;

        int[][] dp = new int[m][n];
        for(int[] arr : dp){
            Arrays.fill(arr, 2001); // 因为最远距离 = m + n <= 2000
        }

        // base case
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(isWater[i][j] == 1){
                    dp[i][j] = 0; // 水域为0
                }
            }
        }

        // 从左上到右下
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(dp[i][j] !=0){
                    if(i > 0){
                        dp[i][j] = Math.min(dp[i-1][j]+1, dp[i][j]); // 上方
                    }
                    if(j > 0){
                        dp[i][j] = Math.min(dp[i][j-1]+1, dp[i][j]); // 左方
                    }
                }
            }
        }

        // 从右下到左上
        for(int i=m-1; i>=0; i--){
            for(int j=n-1; j>=0; j--){
                if(dp[i][j] !=0){
                    if(i < m-1){
                        dp[i][j] = Math.min(dp[i+1][j]+1, dp[i][j]); // 右方
                    }
                    if(j < n-1){
                        dp[i][j] = Math.min(dp[i][j+1]+1, dp[i][j]); // 下方
                    }
                }
            }
        }

        return dp;
    }

    /**
     * 1342. 将数字变成 0 的操作次数
     * 给你一个非负整数 num ，请你返回将它变成 0 所需要的步数。 如果当前数字是偶数，你需要把它除以 2 ；否则，减去 1 。
     * @param num
     * @return
     */
    public static int numberOfSteps(int num) {
        int count = 0;
        if (num == 0) return 0;
        while (num > 0){
            if (num % 2 == 0){
                num /= 2;
            }else {
                num --;
            }
            count ++;
        }
        return count;
    }

    /**
     * 16. 最接近的三数之和
     * @param nums
     * @param target
     * @return
     */
    public static int threeSumClosest(int[] nums, int target) {
        if (nums.length < 3){
            return 0;
        }
        Arrays.sort(nums);
        int minSum = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length - 2; i++) {
            int l = i + 1, r = nums.length - 1;
            while (l < r){
                int sum = nums[i] + nums[l] + nums[r];
                if (Math.abs(target - sum) < Math.abs(target - minSum)) {
                    minSum = sum;// 取距离最近的三数和
                }
                if (sum < target){
                    l++;
                }else if (sum > target){
                    r--;
                }else {
                    return minSum;
                }
            }
        }
        return minSum;
    }

    /**
     * 22. 括号生成 动态规划
     * @param n 生成括号的对数
     * @return 所有可能的并且 有效的 括号组合。
     */
    public static List<String> generateParenthesis(int n) {
        List<String> dp0 = Collections.singletonList("");
        List<String> dp1 = Collections.singletonList("()");
        // 状态集，所有的括号组合情况
        List<List<String>> dp = new ArrayList<>(Arrays.asList(dp0, dp1));
        if (n <= 0){
            return dp0;
        }
        if (n == 1){
            return dp1;
        }
        // 从2开始枚举
        for (int i = 2; i <= n; i++) {
             // dp[n] = "("+dp[p]+")"+dp[q], p + q = n-1
            List<String> t = new ArrayList<>();
            for (int j = 0; j < i; j++) {
                for (String p : dp.get(j)) {
                    for (String q : dp.get(i - j - 1)) {
                        String s = "(" + p + ")" + q;
                        t.add(s);
                    }
                }
            }
            dp.add(t);
        }
        return dp.get(n);
    }

    /**
     * 判断串是否是回文串
     * @return
     */
    public static boolean isPalindrome(String s){
        char[] chars = new char[s.length()];
        for (int i = 0; i < chars.length; i++) {
            chars[i] = s.charAt(i);
        }
//        char[] chars = s.toCharArray();
        for (int l = 0,r = s.length() - 1; l < r; l++,r--) {
            if (chars[l] != chars[r]){
                return false;
            }
        }
        return true;
    }

    /**
     * 33. 搜索旋转排序数组
     * @param nums
     * @param target
     * @return
     */
    public static int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0, right = nums.length-1;
        int mid;
        while (left <= right){
            mid = left + (right - left) / 2;
            // 数组左半部分有序且target属于左半部分，直接在左半部分寻找
            if (target == nums[mid]){
                return mid;
            }
            // 现根据mid确定两边哪边有序
            // 左边有序
            if (nums[left] <= nums[mid]){
                // 属于左边有序区间就在左边找
                if (target < nums[mid] && nums[left] <= target){
                    right = mid - 1;
                }else {
                    left = mid + 1;
                }
            // 右半部分有序
            }else {
                // 属于右边有序区间就在右边找
                if (target > nums[mid] && target <= nums[right]){
                    left = mid + 1;
                }else {
                    right = mid - 1;
                }
            }

        }
        return -1;
    }

    /**
     * 34. 在排序数组中查找元素的第一个和最后一个位置
     * @param nums
     * @param target
     * @return
     */
    public static int[] searchRange(int[] nums, int target) {
        int[] result = new int[]{-1, -1};
        if (nums == null || nums.length == 0){
            return result;
        }
        int left = 0,right = nums.length - 1;
        int mid;
        // 查找左边界
        while (left <= right){
            mid = left + (right - left)/2;
            if (nums[mid] == target){
                if (mid > 0 && nums[mid - 1] == nums[mid]){
                    right = mid - 1;
                }else {
                    result[0] = mid;
                    break;
                }
            }else if (nums[mid] > target){
                right = mid - 1;
            }else {
                left = mid + 1;
            }
        }
        left = 0;
        right = nums.length - 1;
        // 查找右边界
        while (left <= right){
            mid = left + (right - left)/2;
            if (nums[mid] == target){
                if (mid < nums.length-1 && nums[mid + 1] == nums[mid]){
                    left = mid + 1;
                }else {
                    result[1] = mid;
                    break;
                }
            }else if (nums[mid] > target){
                right = mid - 1;
            }else {
                left = mid + 1;
            }
        }
        return result;
    }

    /**
     * 153. 寻找旋转排序数组中的最小值
     * @param nums
     * @return
     */
    public static int findMin(int[] nums) {
        if (nums == null || nums.length == 0){
            return -1;
        }
        int numRight = nums[nums.length - 1];
        int left = 0, right = nums.length - 1;
        int mid;
        while (left <= right){
            mid = left + (right - left) / 2;
            // 说明最小值在mid左边
            if (nums[mid] < numRight){
                right = mid - 1;
            // 说明最小值在mid右边
            }else if (nums[mid] > numRight){
                left = mid + 1;
            }else {
                return nums[mid];
            }
        }
        return nums[left];
    }

    /**
     * 1414. 和为 K 的最少斐波那契数字数目
     * k 一定能减到0
     * 这个快因为不用再在结果集里循环找结果，找第一个大的稍微慢点，但是找下一个小的就会快很多， while (b <= k)次数并不多，
     * 且空间是o(1)
     * 递归方式
     * @param k
     * @return
     */
    public static int findMinFibonacciNumbers(int k) {
        // 递归出口，k=0结束
        if (k == 0) return 0;
        // 求小于
        int a = 1,b = 1;
        while (b <= k){
            int c = a + b;
            a = b;
            b = c;
        }
        // a 为小于等于 k 的最大值，每次计数+1
        return findMinFibonacciNumbers(k - a) + 1;
    }

    /**
     * 48. 旋转图像
     * @param matrix
     */
    public static void rotate(int[][] matrix) {
        int length = matrix.length;
        // 对角线翻转
        for (int i = 1; i < length; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        // 左右翻转
        int l = 0, r = length - 1;
        while (l < r){
            int temp;
            for (int i = 0; i < length; i++) {
                temp = matrix[i][l];
                matrix[i][l] = matrix[i][r];
                matrix[i][r] = temp;
            }
            l++;
            r--;
        }
    }

    /**
     * 45. 跳跃游戏 II
     * @param nums
     * @return
     */
    public static int jump(int[] nums) {
        int step = 0;// 步数
        int maxPos = 0;// 可以跳跃的最大距离
        int end = 0;// 一次跳跃可以到达的最大距离
        for (int i = 0; i < nums.length - 1; i++) {
            maxPos = Math.max(maxPos, i + nums[i]);
            if (i == end){
                end = maxPos;
                step ++;
                // 如果下一条已经可以到达终点直接返回即可
                if (maxPos >= nums.length - 1){
                    return step;
                }
            }
        }
        return step;
    }

    /**
     * 55. 跳跃游戏
     * @param nums
     * @return
     */
    public static boolean canJump(int[] nums) {
        // 如果就一个数就已经起跳成功了
        if (nums.length == 1){
            return true;
        }
        int maxPos = 0;
        int end = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            // 每次循环更新最大跳跃位置
            maxPos = Math.max(maxPos, i + nums[i]);
            // i == end 时开始下一次起跳
            if (i == end){
                // 更新下一条的end位置
                end = maxPos;
                // 如果要起跳的位置超过或等于最终位置则断定可以到达
                if (maxPos >= nums.length - 1){
                    return true;
                }
            }
            // end 更新失败意味着后面的都无法继续进行下去了
            if (i > end){
                return false;
            }
        }
        // 循环走完都到不了就真的到不了
        return false;
    }

    /**
     * 81. 搜索旋转排序数组 II
     * 类比于 I 数组中包含着重复元素
     * 如果所有元素都重复且无法匹配target则时间复杂度增至o(n)
     * @param nums 旋转数组
     * @param target 目标值
     * @return
     */
    public static boolean search2(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int mid;
        while (left <= right){
            mid = left + (right - left) / 2;
            // 找到了直接返回
            if (nums[mid] == target){
                return true;
            }
            if (nums[left] == nums[mid]){
                // 该情况下出现了数组元素重复，让left+1规避重复元素
                left++;
                // 左边有序
            }else if (nums[left] < nums[mid]){
                if (target >= nums[left] && target < nums[mid]){
                    // nums[left]<= target < nums[mid]在左边找
                    right = mid - 1;
                }else {
                    // 否则在右边找
                    left = mid + 1;
                }
                // 右边有序
            }else {
                if (target <= nums[right] && target > nums[mid]){
                    // nums[mid] < target <= nums[right]在右边找
                    left = mid + 1;
                }else {
                    // 否则在左边找
                    right = mid - 1;
                }
            }
        }
        return false;
    }

    /**
     * 96. 不同的二叉搜索树
     * 动态规划
     * dp[n] = dp[0]*dp[n-1] + dp[1]*dp[n-2] + ... + dp[i]*dp[n-i] + ... + dp[n-1]*dp[0]
     * @param n
     * @return
     */
    public static int numTrees(int n) {
        // dp[n]表示n个结点的树种类有多少个
        int[] dp = new int[n+1];
        dp[0] = 1;// 空数种类只有一
        dp[1] = 1;// 只有一个根节点的树种类只有一
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }
        return dp[n];
    }

    /**
     * 80. 删除有序数组中的重复项 II
     * @param nums
     * @return
     */
    public static int removeDuplicates(int[] nums) {
        return removeDuplicatesKeepK(nums,2);
    }

    /**
     * 删除排序数组的重复元素，保留K个
     * @param nums 排序数组
     * @param k 保留K个
     * @return
     */
    public static int removeDuplicatesKeepK(int[] nums, int k){
        // 待放元素位置
        int i = 0;
        // 遍历数组，每次判断当前元素是否加入放置位置
        for (int num : nums) {
            // i < k 初始K个位置可以直接放
            // 如果num和待放置的元素的倒数第二个位置元素相等则无法加入
            if (i < k || nums[i-k] != num)
                // 放置元素，下标后移
                nums[i++] = num;
        }
        // i 即新数组长度
        return i;
    }

    /**
     * 1405. 最长快乐字符串
     * 贪心 + 大顶堆
     * @param a
     * @param b
     * @param c
     * @return
     */
    public static String longestDiverseString(int a, int b, int c) {
        // int[0] 存放a,b,c对应的0，1，2，注意 0+'a' = 'a', 1 + 'a' = 'b', 2 + 'a' = 'c'
        PriorityQueue<int[]> q = new PriorityQueue<>((o1, o2) -> o2[1]-o1[1]);// o2[1]-o1[1]表示大顶堆
        if (a > 0) q.add(new int[]{0, a});
        if (b > 0) q.add(new int[]{1, b});
        if (c > 0) q.add(new int[]{2, c});

        StringBuilder sb = new StringBuilder();

        // 贪心策略，每次选个最大的用来构建
        while (!q.isEmpty()){
            int[] cur = q.poll();
            int length = sb.length();
            // 如果已拼接字符串后两位和出队元素属于同一种则出下一个堆元素进行比较
            if (length >= 2 && cur[0] == sb.charAt(length - 1) - 'a' && cur[0] == sb.charAt(length - 2) - 'a'){
                // 如果cur已经是最后一个且不满足最后拼接条件了循环就结束了
                if (q.isEmpty()) break;
                int[] next = q.poll();
                sb.append((char) (next[0] + 'a'));
                if (--next[1] > 0){
                    // 还有剩的再放回去
                    q.add(next);
                }
                q.add(cur);// 把cur再放回去
            }else {

                if (--cur[1] > 0){
                    // 还有剩的再放回去
                    q.add(cur);
                }
            }
        }
        return sb.toString();
    }

    /**
     * 31. 下一个排列
     * @param nums
     */
    public static void nextPermutation(int[] nums) {
        if (nums.length == 1) return;
        int i = nums.length - 1;
        // 倒数找第一组(i-1,i),(i-1,i)满足nums[i-1] < nums[i]
        while (i > 0 && nums[i] <= nums[i-1]) i--;
        // i - 1 >= 0时意味着存在 nums[i] > nums[i-1] 的数对
        if (i > 0){
            int j = nums.length-1;
            // 倒数找第一个大于 nums[i-1] 的元素与他做交换
            while (j >= 0 && nums[j] <= nums[i-1]) j--;
            swap(nums, j, i-1);
        }
        // [i, end] 此时一定逆序，将其翻转
        reverse(nums, i);
    }

    public static void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public static void reverse(int[] nums, int start){
        int l = start, r = nums.length - 1;
        while (l < r){
            swap(nums, l , r);
            l++;
            r--;
        }
    }

    /**
     * 540. 有序数组中的单一元素
     * 不用 r = mid -1, l = mid 是因为这样走会丢失掉单一元素位于mid的情况，r的最终位置就是目标值
     * @param nums
     * @return
     */
    public static int singleNonDuplicate(int[] nums) {
        int l = 0, r = nums.length - 1;
        int mid;
        while (l < r){
            mid = l + (r - l) / 2;
            // 中点位于偶数位，偶数位与它所在后一位比较，如果相同，左半部分元素未发生失序，在右半部分找
            if (mid % 2 == 0){
                // 右半部分完好在左边找
                if (mid < nums.length - 1 && nums[mid] == nums[mid+1]){
                    l = mid + 1;
                }else {
                    r = mid;
                }
                // 中点位于奇数位，奇数位位与它所在前一位比较，如果相同，左半部分元素未发生失序，在右半部分找
            }else {
                //
                if (mid > 0 && nums[mid] == nums[mid-1]){
                    l = mid + 1;
                }else {
                    r = mid;
                }
            }
        }
        return nums[r];
    }
//    /**
//     * 540. 有序数组中的单一元素
//     * @param nums
//     * @return
//     */
//    public static int singleNonDuplicate(int[] nums) {
//        int l = 0, r = nums.length - 1;
//        int mid;
//        while (l < r){
//            mid = l + (r - l) / 2;
//            // 右半部分完好在左边找
//            if (nums[mid] == nums[mid^1])
//                l = mid + 1;
//            else
//                r = mid;
//        }
//        return nums[r];
//    }

    /**
     * 74. 搜索二维矩阵
     * @param matrix
     * @param target
     * @return
     */
    public static boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int l = 0, r = m * n - 1;
        int mid;
        while (l <= r){
            mid = l + (r - l) / 2;
            int val = matrix[mid / n][mid % n];
            if (val == target){
                return true;
            }else if (val > target){
                r = mid - 1;
            }else {
                l = mid + 1;
            }
        }
        return false;
    }

    /**
     * 378. 有序矩阵中第 K 小的元素
     * @param matrix
     * @param k 总的第k小
     * @return
     */
    public static int kthSmallest(int[][] matrix, int k) {
        int left = matrix[0][0], right = matrix[matrix.length-1][matrix.length-1];
        // mid 是用来猜的数，猜的mid不一定真的是矩阵元素
        int mid;
        // 当到达边界条件时，left = right 即为最终答案，并非是mid准确无误的成为了第k大的元素
        while (left < right){
            mid = left + (right - left) / 2;
            if (check(matrix, mid, k)){
                // right 用来收缩边界，当比mid小的元素多于k时表明mid猜大了，将right收缩一下继续猜
                right = mid;
            }else {
                // left 用来确认最终答案 当比mid小的元素少于k时表明mid猜小了，mid + 1 不停地进行试探
                left = mid + 1;
            }
        }
        return left;
    }

    /**
     * 检验小于等于mid的元素数量是否比k大
     * @param matrix
     * @param mid
     * @param k
     * @return
     */
    private static boolean check(int[][] matrix, int mid, int k){
        // 初始化比mid值小的元素数量和
        int sum = 0;
        int i = matrix.length - 1, j = 0;
        while (i >= 0 && j < matrix[0].length){
            if (matrix[i][j] <= mid){
                sum += i + 1;
                j++;
            }else {
                i--;
            }
        }
        return sum >= k;
    }

    /**
     * 63. 不同路径 II
     * @param obstacleGrid
     * @return
     */
    public static int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
           if (obstacleGrid[i][0] == 0)
               dp[i][0] = 1;
           else {
              while (i < m){
                  dp[i][0] = 0;
                  i++;
              }
           }
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 0)
                dp[0][i] = 1;
            else {
                while (i < n){
                    dp[0][i] = 0;
                    i++;
                }
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 0){
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];
    }

//    /**
//     * 91. 解码方法
//     * 动态规划思想
//     * dp[i] = dp[i-1] + (if (true) dp[i-2])
//     * @param s
//     * @return
//     */
//    public static int numDecodings(String s) {
//        int n = s.length();
//        char[] chars = s.toCharArray();
//        int[] dp = new int[n + 1];// dp[0] 0-n
//        // dp[0]可以假当是空串，dp[i]为以字符i为止的可以编码的数量，相当于哨兵
//        dp[0] = 1;
//        // 由第一个字符到第n个字符依次推演，不能将第一个字符当作dp起始条件做推演，这样结果是不对的
//        for (int i = 1; i <= n; i++) {
//            if (chars[i-1] != '0'){
//                dp[i] += dp[i-1];
//            }
//            if (i > 1 && chars[i-2] != '0' && (chars[i-2] - '0') * 10 + (chars[i-1] - '0') <= 26){
//                dp[i] += dp[i-2];
//            }
//        }
//        return dp[n];
//    }
    /**
     * 91. 解码方法
     * 动态规划思想 空间优化
     * c = a + if true : b
     * @param s
     * @return
     */
    public static int numDecodings(String s) {
        int n = s.length();
        char[] chars = s.toCharArray();
        // dp[i-2], dp[i-1], dp[i] 由于初始下标-1不存在，可以直接初始化0，b = 1 相当于初始dp[0] = 1，
        // c 初始化0是为了避免边界条件，万一循环不走将无值返回编译报错
        int a = 0, b = 1, c = 0;
        // 由第一个字符到第n个字符依次推演，不能将第一个字符当作dp起始条件做推演，这样结果是不对的
        for (int i = 1; i <= n; i++) {
            c = 0;
            if (chars[i-1] != '0'){
                c += b;
            }
            if (i > 1 && chars[i-2] != '0' && (chars[i-2] - '0') * 10 + (chars[i-1] - '0') <= 26){
                c += a;
            }
            a = b;
            b = c;
        }
        return c;
    }

    /**
     * 64. 最小路径和
     * @param grid
     * @return
     */
    public static int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        // 初始化列
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i-1] + grid[0][i];
        }
        // 初始化行
        for (int j = 1; j < m; j++) {
            dp[j][0] = dp[j-1][0] + grid[j][0];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = grid[i][j] + Math.min(dp[i-1][j],dp[i][j-1]);
            }
        }
        return dp[m-1][n-1];
    }

//    /**
//     * 198. 打家劫舍
//     * @param nums
//     * @return
//     */
//    public static int rob(int[] nums) {
//        int n = nums.length;
//        if (n == 1) return nums[0];
//        int[] dp = new int[n];
//        dp[0] = nums[0];
//        dp[1] = Math.max(nums[0], nums[1]);
//
//        for (int i = 2; i < n; i++) {
//            // 尝试判断是否偷当前这一户，对比偷的收益来决定
//            // 如果收益超过了上一家的收益，可以偷
//            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
//        }
//        return dp[n-1];
//    }

    /**
     * 213. 打家劫舍 II
     * @param nums
     * @return
     */
    public static int rob2(int[] nums) {
        // 涉及两种情况，如果偷了第一户就不能偷到最后一户，如果第一户不偷就可以顺理成章偷到最后一户
        int n = nums.length;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n - 1; i++) {
            dp[i] = Math.max(dp[i-2] + nums[i], dp[i-1]);
        }
        int t = dp[n - 2];
        Arrays.fill(dp,0);
        dp[1] = nums[1];
        dp[2] = Math.max(nums[1], nums[2]);
        for (int i = 3; i < n; i++) {
            dp[i] = Math.max(dp[i-2] + nums[i], dp[i-1]);
        }
        return Math.max(dp[n - 1], t);
    }

    /**
     * 198. 打家劫舍
     * 空间优化
     * @param nums
     * @return
     */
    public static int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        int a = nums[0], b = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            int c = b;
            // 尝试判断是否偷当前这一户，对比偷的收益来决定
            // 如果收益超过了上一家的收益，可以偷
            b = Math.max(a + nums[i], b);
            a = c;
        }
        return b;
    }

    /**
     * 29. 两数相除
     * @param dividend
     * @param divisor
     * @return
     */
    public static int divide(int dividend, int divisor) {
        // 边界条件 -2^32 / -1 = 2^32 越界
        if (dividend == Integer.MIN_VALUE && divisor == -1) return Integer.MAX_VALUE;
        // limit 防止翻倍后发生越界
        int limit = Integer.MIN_VALUE >> 1;
        boolean sign;
        // 先确认结果的符号，然后映射成负数处理
        sign = (dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0);
        if (dividend > 0) dividend = -dividend;
        if (divisor > 0) divisor = -divisor;
        int result = 0;
        while (dividend <= divisor){
            int t = divisor, count = 1;
            // divisor 每次呈指数倍翻倍逼近 dividend，翻倍的数据记在t中，翻倍的次数记在count中
            while (t >= limit && count >= limit && t > dividend - t){
                t += t;
                count += count;
            }
            dividend -= t;// dividend减去最大幅度逼近的除数开启下一波匹配
            result += count;// 将翻倍的次数累加，所有次数加在一起就是商
        }
        return sign ? result : -result;// 根据符号决定商的符号
    }

    /**
     * 54. 螺旋矩阵
     * @param matrix
     * @return
     */
    public static List<Integer> spiralOrder(int[][] matrix) {
        int top = 0, left = 0, bottom = matrix.length - 1, right = matrix[0].length - 1;
        List<Integer> result = new ArrayList<>();
        while (true){
            // 沿着上界从左向右走
            for (int i = left; i <= right; i++) result.add(matrix[top][i]);
            if (++top > bottom) break;// 走完上界减一 上界减完越界退出
            // 沿着右界从上向下走
            for (int i = top; i <= bottom; i++) result.add(matrix[i][right]);
            if (--right < left) break;// 走完右界减一 越界退出
            for (int i = right; i >= left; i--) result.add(matrix[bottom][i]);
            if (--bottom < top) break;
            for (int i = bottom; i >= top; i--) result.add(matrix[i][left]);
            if (++left > right) break;
        }
        return result;
    }

//    /**
//     * 75. 颜色分类
//     * 单指针法
//     * @param nums
//     */
//    public static void sortColors(int[] nums) {
//        // 指向已经归纳好的一种类型的下一个元素的位置
//        int ptr = 0;
//        // 先归纳出0的元素
//        for (int i = 0; i < nums.length; i++) {
//            if (nums[i] == 0){
//                int temp = nums[i];
//                nums[i] = nums[ptr];
//                nums[ptr] = temp;
//                ptr++;
//            }
//
//        }
//        for (int i = ptr; i < nums.length; i++) {
//            if (nums[i] == 1){
//                int temp = nums[i];
//                nums[i] = nums[ptr];
//                nums[ptr] = temp;
//                ptr++;
//            }
//        }
//    }
//    /**
//     * 75. 颜色分类
//     * 双指针法
//     * @param nums
//     */
//    public static void sortColors(int[] nums) {
//        // p0 归纳 0，p1 归纳 1
//        int p0 = 0, p1 = 0;
//        for (int i = 0; i < nums.length; i++) {
//            if (nums[i] == 0){
//                int temp = nums[p0];
//                nums[p0] = nums[i];
//                nums[i] = temp;
//                // 这个过程会将1换出去 需要再次和p1交换换回来
//                if (p0 < p1){
//                    temp = nums[p1];
//                    nums[p1] = nums[i];
//                    nums[i] = temp;
//                }
//                p0++;
//                p1++;
//            }
//            if (nums[i] == 1){
//                int temp = nums[p1];
//                nums[p1] = nums[i];
//                nums[i] = temp;
//                p1++;
//            }
//        }
//
//    }
    /**
     * 75. 颜色分类
     * 双指针法 排0，2
     * @param nums
     */
    public static void sortColors(int[] nums) {
        // p0 归纳 0，p1 归纳 1
        int p0 = 0, p2 = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0){
                int temp = nums[p0];
                nums[p0] = nums[i];
                nums[i] = temp;
                p0++;
            }
            // i 到达 p2 就停止
            if (i <= p2 && nums[i] == 2){
                int temp = nums[p2];
                nums[p2] = nums[i];
                nums[i] = temp;
                p2--;
            }
        }
    }

    /**
     * 89. 格雷编码
     * n 位格雷编码 为n-1位取反+2^n
     * @param n
     * @return
     */
    public static List<Integer> grayCode(int n) {
        List<Integer> result = new ArrayList<>();
        result.add(0);
        int head = 1;
        for (int i = 0; i < n; i++) {
            for (int j = result.size() - 1; j >= 0 ; j--) {
                result.add(result.get(j) + head);
            }
            head <<= 1;
        }
        return result;
    }

    /**
     * 128. 最长连续序列
     * @param nums
     * @return
     */
    public static int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int maxCount = 0;
        for (Integer num : set) {
            // 确保不重复选取某些数字，当 num - 1 存在时直接跳过
            if (!set.contains(num - 1)){
                int count = 1;
                int n = num + 1;
                while (set.contains(n)){
                    count++;
                    n++;
                }
                maxCount = Math.max(maxCount, count);
            }
        }
        return maxCount;
    }

//    /**
//     * 97. 交错字符串
//     * s1的前i个串和s2的前j个串是否可以构成s3的前i+j个串
//     * dp[i,j] = dp[i-1][j] && s1[i] == s3[i]
//     * @param s1
//     * @param s2
//     * @param s3
//     * @return
//     */
//    public boolean isInterleave(String s1, String s2, String s3) {
//        boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];
//        dp[0][0] = true;
//        for (int i = 0; i < s1.length(); i++) {
//            for (int j = 0; j < s2.length(); j++) {
//                int p = i + j - 1;
//                if (i > 0)
//                    dp[i][j] =
//            }
//        }
//    }

    /**
     * 59. 螺旋矩阵 II
     * @param n
     * @return
     */
    public static int[][] generateMatrix(int n) {
        int[][] matrix = new int[n][n];
        int left = 0, top = 0, right = n - 1, bottom = n - 1;
        int count = 0;
        while (true){
            for (int i = left; i <= right; i++) matrix[top][i] = ++count;
            if (++top > bottom) break;
            for (int i = top; i <= bottom; i++) matrix[i][right] = ++count;
            if (--right < left) break;
            for (int i = right; i >= left; i--) matrix[bottom][i] = ++count;
            if (--bottom < top) break;
            for (int i = bottom; i >= top; i--) matrix[i][left] = ++count;
            if (++left > right) break;
        }
        return matrix;
    }

//    /**
//     * 136. 只出现一次的数字
//     * 0 异或 任何数都是其本身
//     * @param nums
//     * @return
//     */
//    public static int singleNumber(int[] nums) {
//        int result = nums[0];
//        for (int i = 1; i < nums.length; i++) {
//            result ^= nums[i];
//        }
//        return result;
//    }

    /**
     * 137. 只出现一次的数字 II
     * @param nums
     * @return
     */
    public static int singleNumber(int[] nums) {
        // 长度为32的常量空间
        int[] counts = new int[32];
        // 依次累计求每个位的和
        for (int num : nums) {
            for (int j = 0; j < 32; j++) {
                // 每次获取num的最低位累加至对应桶
                counts[j] += num & 1;
                num >>>= 1;
            }
        }
        int result = 0, m = 3;
        for (int i = 0; i < 32; i++) {
            result <<= 1;
            // 从高位向低位依次还原，得到一位还原一位
            result |= counts[31 - i] % m;
        }
        return result;
    }

    /**
     * 209. 长度最小的子数组
     * @param target
     * @param nums
     * @return
     */
    public static int minSubArrayLen(int target, int[] nums) {
        // high++ 入队， low++ 出队
        int low = 0, high = 0, sum = 0, min = Integer.MAX_VALUE;
        while (high < nums.length){
            sum += nums[high++];
            while (sum >= target){
                min = Math.min(high - low, min);
                sum -= nums[low++];
            }
        }
        return min == Integer.MAX_VALUE ? 0 : min;
    }

    /**
     * 172. 阶乘后的零
     * 5 * 2 = 10, 2 的数量远大于 5 的数量
     * 有多少个5出现结尾就有多少个0出现
     * 每5个数相乘结果便会出现一个5，每25个数之间会出来2个5，每125个数之间会出来3个5......
     * @param n
     * @return
     */
    public static int trailingZeroes(int n) {
        int count = 0;
        while (n > 0){
            // 除去一轮计算下一轮
            n /= 5;
            // 相隔5的5的数量
            count += n;
        }
        return count;
    }

    /**
     * 167. 两数之和 II - 输入有序数组
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum2(int[] numbers, int target) {
        int i = 0, j = numbers.length - 1;
        while (i < j){
            int sum = numbers[i] + numbers[j];
            if (sum < target){
                i++;
            }else if (sum > target){
                j--;
            }else {
                return new int[]{i + 1, j + 1};
            }
        }
        return new int[]{-1, -1};
    }

    /**
     * 56. 合并区间
     * @param intervals
     * @return
     */
    public static int[][] merge(int[][] intervals) {
        if (intervals.length == 1) return intervals;
        LinkedList<int[]> result = new LinkedList<>();
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        result.addLast(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            int[] last = result.getLast();
            // 第一个元素相等时将第二个元素最大的数组插入末尾
            if (last[0] == intervals[i][0]) {
                result.removeLast();
                result.addLast(new int[]{last[0], Math.max(intervals[i][1], last[1])});
            } else if (last[0] < intervals[i][0]) {
                if (last[1] >= intervals[i][0] ) {
                    // [0,3] [1,9] || [0,3] [3,9]
                    if (last[1] < intervals[i][1]){
                        result.removeLast();
                        // [0, 9]
                        result.addLast(new int[]{last[0], intervals[i][1]});
                    }
                    // [0,9] [2,5] -> 跳过
                } else {
                    // [0,3] [4,9] -> 直接加
                    result.addLast(intervals[i]);
                }
            }
        }
        return result.toArray(new int[result.size()][]);
    }

//    /**
//     * 43. 字符串相乘
//     * @param num1
//     * @param num2
//     * @return
//     */
//    public static String multiply(String num1, String num2) {
//        if (num1.equals("0") || num2.equals("0")) return "0";
//        StringBuilder result = new StringBuilder();
//        for (int i = num2.length() - 1; i >= 0; i--) {
//            int carry = 0, digit;
//            StringBuilder temp = new StringBuilder();
//            // 补0
//            for (int j = 0; j < num2.length() - i - 1; j++) {
//                temp.append(0);
//            }
//            for (int j = num1.length() - 1; j >= 0 || carry > 0; j--) {
//                int x = j < 0 ? 0 : num1.charAt(j) - '0';// 单独处理j < 0 的情况
//                int product = x * (num2.charAt(i) - '0') + carry;
//                carry = product / 10;
//                digit = product % 10;
//                temp.append(digit);
//            }
//            StringBuilder reverse = temp.reverse();
//            String s = addStrings(reverse.toString(), result.toString());
//            result = new StringBuilder(s);
//
//        }
//        return result.toString();
//    }

    /**
     * 415. 字符串相加
     * @param num1
     * @param num2
     * @return
     */
    public static String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        for (int i = num1.length() - 1, j = num2.length() - 1;
             i >= 0 || j >= 0 || carry > 0;
             i--, j--) {
            int x = i < 0 ? 0 : num1.charAt(i) - '0';// 取 i
            int y = j < 0 ? 0 : num2.charAt(j) - '0';// 取 j
            int sum = x + y + carry;
            carry = sum / 10;
            sb.append(sum % 10);
        }
        return sb.reverse().toString();
    }

    /**
     * 43. 字符串相乘
     * 竖式优化
     * @param num1
     * @param num2
     * @return
     */
    public static String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) return "0";
        StringBuilder result = new StringBuilder();
        int m = num1.length();
        int n = num2.length();
        int[] container = new int[m + n];// m 位数 * n 位数 最多 m * n 位数
        for (int i = n - 1; i >= 0; i--) {
            int x = num2.charAt(i) - '0';
            for (int j = m - 1; j >= 0; j--) {
                int y = num1.charAt(j) - '0';
                // num1[i] x num2[j] 的结果为 tmp(位数为两位，"0x","xy"的形式)，其第一位位于 res[i+j]，第二位位于 res[i+j+1]
                int sum = x * y + container[i + j + 1];
                container[i + j + 1] = sum % 10;
                container[i + j] += sum / 10;
            }
        }
        for (int i = 0; i < container.length; i++) {
            if (i == 0 && container[0] == 0) continue;
            result.append(container[i]);
        }

        return result.toString();
    }

}
