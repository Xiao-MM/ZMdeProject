package algorithm;

import org.omg.CORBA.MARSHAL;

import java.util.*;
import java.util.stream.Collectors;

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
    public static int singleNumber2(int[] nums) {
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

    /**
     * 152. 乘积最大子数组
     * 动态规划
     * @param nums
     * @return
     */
    public static int maxProduct(int[] nums) {
        // max 最终结果， maxi 到达第i个元素所能得到的最大乘积，mini 到达第i个元素所能得到的最小乘积
        int max = Integer.MIN_VALUE, maxi = 1, mini = 1;
        for (int num : nums) {
            // 如果num小于0 maxi 和 mini 互换以更新新的最大和最小值
            if (num < 0) {
                int temp = maxi;
                maxi = mini;
                mini = temp;
            }
            // 更新最大值
            maxi = Math.max(num, maxi * num);
            // 更新最小值
            mini = Math.min(num, mini * num);
            // 更新结果
            max = Math.max(max, maxi);
        }
        return max;
    }

    /**
     * 189. 轮转数组
     * @param nums
     * @param k
     */
    public static void rotate(int[] nums, int k) {
        int length = nums.length;
        k %= length;
        if (k == 0) return;
        reverse(nums, 0, length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, length - 1);


    }

    /**
     * 189. 轮转数组
     * @param nums
     * @param k
     */
    public static void rotate2(int[] nums, int k) {
        int length = nums.length;
        k %= length;
        if (k == 0) return;
        // 后半部分数组长度
        int behindLen = length - k;
        int[] temp = new int[behindLen];
        System.arraycopy(nums, 0, temp, 0, behindLen);
        System.arraycopy(nums, behindLen, nums, 0, k);
        System.arraycopy(temp, 0, nums, k, behindLen);
    }

    /**
     * 旋转数组
     * @param nums
     * @param left
     * @param right
     */
    public static void reverse(int[] nums, int left, int right){
        while (left < right){
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            left ++;
            right--;
        }
    }

    /**
     * 166. 分数到小数
     * @param numerator
     * @param denominator
     * @return
     */
    public static String fractionToDecimal(int numerator, int denominator) {
        // -2^32 / -1 溢出
        long a = numerator, b = denominator;
        if (a % b == 0) return String.valueOf(a / b);
        StringBuilder sb = new StringBuilder();
        // 追加 -
        if (a * b < 0) sb.append("-");
        a = Math.abs(a);
        b = Math.abs(b);
        // 先获取其整数部分
        sb.append(a / b).append(".");
        // 拿到余数
        a %= b;
        Map<Long, Integer> map = new HashMap<>();
        while (a != 0){
            // 将余数所在位置加入map
            map.put(a, sb.length());
            a *= 10;
            // 追加除数
            sb.append(a / b);
            // 再次获取余数
            a %= b;
            // 如果出现了余数则不必再进行计算了
            if (map.containsKey(a)){
                Integer pos = map.get(a);
                return String.format("%s(%s)", sb.substring(0, pos), sb.substring(pos));
            }
        }
        return sb.toString();
    }

    /**
     * 162. 寻找峰值
     * @param nums
     * @return
     */
    public static int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        int mid;
        // 长度为1时 while (l < r) 直接不走了
        while (l < r){
            mid = l + r >> 1;
            // mid 元素 > mid + 1 元素，在左边存在峰值，
            if (nums[mid] > nums[mid + 1])
                // mid 可能为峰值，比如 num[mid - 1] < num[mid]，mid就是峰值
                r = mid;
            else
                // nums[mid] < nums[mid + 1] 的情况下mid就不一定是峰值了，可以担保向右一定会出现峰值
                l = mid + 1;
        }
        // 跳出循环时的r就是最终答案
        return r;
    }

    /**
     * 11. 盛最多水的容器
     * @param height
     * @return
     */
    public static int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int max = 0, water;
        while (l < r){
            water = Math.min(height[l], height[r]) * (r - l);
            max = Math.max(max, water);
            if (height[l] <= height[r])
                l++;
            else
                r--;
        }
        return max;
    }

    /**
     * 12. 整数转罗马数字
     * @param num
     * @return
     */
    public static String intToRoman(int num) {
        Map<Integer, String> map = new HashMap<>();
        map.put(1, "I");
        map.put(4, "IV");
        map.put(5, "V");
        map.put(9, "IX");
        map.put(10, "X");
        map.put(40, "XL");
        map.put(50, "L");
        map.put(90, "XC");
        map.put(100, "C");
        map.put(400, "CD");
        map.put(500, "D");
        map.put(900, "CM");
        map.put(1000, "M");
        int[] keys = new int[]{1,4,5,9,10,40,50,90,100,400,500,900,1000};
        StringBuilder sb = new StringBuilder();
        for (int i = keys.length - 1; i >= 0; i--) {
            if (num < keys[i]) continue;
            int remainder = num / keys[i];
            for (int j = 0; j < remainder; j++) {
                sb.append(map.get(keys[i]));
            }
            num %= keys[i];
        }
        return sb.toString();
    }

//    String[] thousands = {"", "M", "MM", "MMM"};
//    String[] hundreds  = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
//    String[] tens      = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
//    String[] ones      = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
//
//     /**
//     * 12. 整数转罗马数字
//     * @param num
//     * @return
//     */
//    public String intToRoman(int num) {
//        return thousands[num / 1000] +
//                hundreds[num % 1000 / 100] +
//                tens[num % 100 / 10] +
//                ones[num % 10];
//    }

    /**
     * 71. 简化路径
     * @param path
     * @return
     */
    public static String simplifyPath(String path) {
        Deque<String> stack = new ArrayDeque<>();
        int length = path.length();
        // i 不默认递增，根据逻辑递增
        for (int i = 1; i < length;) {
            // 找到一个不为 / 的字符，i 指向该位置
            if (path.charAt(i) == '/' && ++i >= 0) continue;
            // j 从 i + 1 向后找
            int j = i + 1;
            // j 指向 下一个 /
            while (j < length && path.charAt(j) != '/') j++;
            // 截取 / part /
            String part = path.substring(i, j);
            // part 为 .. 出栈
            if (part.equals("..")){
                if (!stack.isEmpty()) stack.pollLast();
                // part 为 . 跳过，除此以外的字符送入栈内
            }else if (!part.equals(".")){
                stack.addLast(part);
            }
            // i 沿着 j 继续进行
            i = j;
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) sb.append("/").append(stack.pollFirst());
        return sb.length() == 0 ? "/" : sb.toString();
    }

    /**
     * 97. 交错字符串
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public static boolean isInterleave(String s1, String s2, String s3) {
        int len1 = s1.length(), len2 = s2.length(), len3 = s3.length();
        if (len1 + len2 != len3) return false;
        // s1的前i个和s2的前j个是否可以构成s3的前i+j个
        boolean[][] dp = new boolean[len1 + 1][len2 + 1];
        dp[0][0] = true;
        // 初始化第一列
        for (int i = 1; i <= len1; i++){
            if (dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1)) dp[i][0] = true;
        }
        // 初始化第一行
        for (int i = 1; i <= len2; i++){
            if (dp[0][i - 1] && s2.charAt(i - 1) == s3.charAt(i - 1)) dp[0][i] = true;
        }
        // 更新所有状态数组
        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++)
                // 注意 s1,s2,s3的字符下标是从0 -> length()-1, 状态矩阵的下标是0 -> length， 实际扫描到字母是从1开始计数的
                dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
        }
        return dp[len1][len2];
    }

//    /**
//     * 49. 字母异位词分组
//     * @param strs
//     * @return
//     */
//    public static List<List<String>> groupAnagrams(String[] strs) {
//        return new ArrayList<>(Arrays.stream(strs).collect(Collectors.groupingBy(s -> {
//            char[] chars = s.toCharArray();
//            Arrays.sort(chars);
//            return new String(chars);
//        })).values());
//    }


    /**
     * 49. 字母异位词分组
     * @param strs
     * @return
     */
    public static List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            String key = encode(str);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<>(map.values());
    }

    /**
     * 计数编码
     * @param s
     * @return
     */
    private static String encode(String s){
        char[] chars = new char[26];
        // 统计每个字母出现的次数，并将其作为字母异位词的判别依据
        for (int i = 0; i < s.length(); i++) chars[s.charAt(i) - 'a']++;
        return String.valueOf(chars);
    }

    static class WordBreak {
        /**
         * 139. 单词拆分
         * 递归回溯+记忆化数组
         * @param s
         * @param wordDict
         * @return
         */
        public static boolean wordBreak(String s, List<String> wordDict) {
            // 记忆数组，记录某个位置起始的递归调用是否成功或者失败，若下次有别的调用走到这里可以直接拿结果用
            Boolean[] isHappened = new Boolean[s.length()];
            return dfs(s, 0, isHappened, wordDict);
        }

        private static boolean dfs(String s, int begin, Boolean[] isHappened, List<String> wordDict){
            // 前面成功判定返回
            if (begin == s.length()) return true;
            // 算过了
            if (isHappened[begin] != null) return isHappened[begin];
            for (int i = begin + 1; i <= s.length(); i++) {
                String partStr = s.substring(begin, i);
                // 左边匹配且剩余部分也匹配返回
                // dfs(s, i, wordDict) 只有 wordDict.contains(partStr) 满足才会继续进行下去，当剩下部分不足以匹配 begin == s.length()就不会执行
                if (wordDict.contains(partStr) && dfs(s, i, isHappened, wordDict)){
                    isHappened[begin] = true;// begin开始的串判定已经计算过了，而且判定成功
                    return true;
                }
            }
            // 循环走完都没有成功判定结果返回false
            // begin开始的串判定已经计算过了，而且判定失败
            isHappened[begin] = false;
            return false;
        }
    }

//    /**
//     * 139. 单词拆分
//     * 动态规划
//     * @param s
//     * @param wordDict
//     * @return
//     */
//    public static boolean wordBreak(String s, List<String> wordDict) {
//        // 状态数组，dp[i] 表示s的前i个字符构成的子串是否可以完成拆分
//        boolean[] dp = new boolean[s.length() + 1];
//        // 默认空串是可以达成要求的，这里它的存在单纯为了解决边界情况
//        dp[0] = true;
//        // 状态转移为dp[i] = dp[j] && check(j, i)，j属于[0，i-1]
//        // 前j个元素s[0,j-1]可以由单词组成的情况下，即dp[j] == true时 s[j,i-1] 也能由单词组成，此时dp[i] = true
//        // 注意dp范围[0, n]，s[0, n-1]
//        for (int i = 1; i <= s.length(); i++) {
//            for (int j = 0; j < i; j++) {
//                if (dp[j] && wordDict.contains(s.substring(j, i))){
//                    dp[i] = true;
//                    break;
//                }
//
//            }
//        }
//        return dp[s.length()];
//    }

    /**
     * 139. 单词拆分
     * 动态规划优化
     * @param s
     * @param wordDict
     * @return
     */
    public static boolean wordBreak(String s, List<String> wordDict) {
        int maxWordLength = 0;
        int minWordLength = Integer.MAX_VALUE;
        for(String word : wordDict){
            maxWordLength = Math.max(maxWordLength, word.length());
            minWordLength = Math.min(minWordLength, word.length());
        }
        boolean[] dp = new boolean[s.length()];
        dp[0] = wordDict.contains(s.substring(0,1));
        for(int i=Math.max(1, minWordLength-1);i<s.length();i++){
            for(int j=minWordLength-1;j<=i && j<=maxWordLength;j++){
                if(dp[i-j] && wordDict.contains(s.substring(i-j+1, i+1))){
                    dp[i] = true;
                    break;
                }
                if(i<maxWordLength && wordDict.contains(s.substring(0, i+1))){
                    dp[i] = true;
                }
            }
        }
        return dp[s.length()-1];
    }

    /**
     * 260. 只出现一次的数字 III
     * @param nums
     * @return
     */
    public static int[] singleNumber(int[] nums) {
        int[] result = new int[2];
        // sum 为最终答案异或值，其余两个相等的元素异或之后和为0
        int sum = 0;
        for (int num : nums) sum ^= num;
        int k = -1;
        for (int i = 31; i >= 0; i--) {
            // 找到sum中两目标值不同的位 k
            // 第k位为1表示result[0]和result[1]的第K位必定不相同
            if((sum >> i & 1) == 1) k = i;
        }
        for (int num : nums) {
            // 通过逐个比对num的第k位将其划分到不同的桶子里
            if ((num >> k & 1) == 1) result[1] ^= num;
            else result[0] ^= num;
        }
        return result;
    }


//    /**
//     * 371. 两整数之和
//     * @param a
//     * @param b
//     * @return
//     */
//    public static int getSum(int a, int b) {
//        int u, v, t = 0, sum = 0;
//        for (int i = 0; i < 32; i++) {
//            // 依次取末位，从后往前加
//            u = (a >> i) & 1;
//            v = (b >> i) & 1;
//            if (u == 1 && v == 1){
//                // 当前位的值取决于进位t，发生进位 t 为 1
//                sum |= (t << i);
//                t = 1;
//            }else if (u == 1 || v == 1){
//                // 当 t = 1 时，当前位值为0，进位 t 仍为 1
//                // 当 t = 0 时，当前位值为1，进位 t 仍为 0
//                sum |= ((t ^ 1) << i);
//            }else {
//                // 当两个值都为0时，值取决于进位 t，进位 t 为 0
//                sum |= (t << i);
//                t = 0;
//            }
//        }
//        return sum;
//    }

    /**
     * 371. 两整数之和
     * 递归
     * 先计算原始的 aa 的和原始 bb 在不考虑进位的情况下的结果，结果为 a ^ b，然后在此基础上，考虑将进位累加进来，累加操作可以递归使用 getSum 来处理。
     * 问题转化为如何求得 aa 和 bb 的进位值。
     * 不难发现，当且仅当 aa 和 bb 的当前位均为 11，该位才存在进位，同时进位回应用到当前位的下一位（左边的一边，高一位），因此最终的进位结果为 (a & b) << 1。
     * 因此，递归调用 getSum(a ^ b, (a & b) << 1) 我们可以得到最终答案。
     * 最后还要考虑，该拆分过程何时结束。
     * 由于在进位结果 (a & b) << 1 中存在左移操作，因此最多执行 3232 次的递归操作之后，该值会变为 00，而 00 与任何值 xx 相加结果均为 xx。
     * @param a
     * @param b
     * @return
     */
    public static int getSum(int a, int b) {
        // a ^ b 为 a ，b 不带进位相加，a & b 为 a,b 可以产生的进位，左移是为了将进位加到下一位上
        return b == 0 ? a : getSum(a ^ b, (a & b) << 1);
    }

    /**
     * 134. 加油站
     * @param gas
     * @param cost
     * @return
     */
    public static int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        // 最小剩余油量，最小剩余油量下标，当前剩余油量
        int minSpare = Integer.MAX_VALUE, minIndex = -1, spare = 0;
        // 从下标 0 开始往后跑
        for (int i = 0; i < n; i++) {
            spare = spare + gas[i] - cost[i];
            // 记录最小剩油量
            if (spare < minSpare) {
                minSpare = spare;
                minIndex = i;
            }
        }
        // 剩余油量大于等于0才有可能走完，从最小剩余油量的下一个位置走起就可以走完全程
        return spare < 0 ? -1 : (minIndex + 1) % n;
    }

    /**
     * 179. 最大数
     * @param nums
     * @return
     */
    public static String largestNumber(int[] nums) {
        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strings[i] = String.valueOf(nums[i]);
        }
        // ab 比上 ba
        Arrays.sort(strings, (o1, o2) -> (o2 + o1).compareTo(o1 + o2));
        if (strings[0].equals("0")) return "0";
        StringBuilder sb = new StringBuilder();
        for (String string : strings) {
            sb.append(string);
        }
        return sb.toString();
    }

//    /**
//     * 187. 重复的DNA序列
//     * @param s
//     * @return
//     */
//    public static List<String> findRepeatedDnaSequences(String s) {
//        List<String> result = new ArrayList<>();
//        Map<String, Integer> map = new HashMap<>();
//        for (int i = 0; i <= s.length() - 10; i++) {
//            String subStr = s.substring(i, i + 10);
//            Integer count = map.getOrDefault(subStr, 0);
//            if (count == 1){
//                result.add(subStr);
//            }
//            map.put(subStr, count + 1);
//        }
//        return result;
//    }

    /**
     * 187. 重复的DNA序列
     * 垃圾 ！ 比上面的还慢
     * 位运算，这样的好处不需要每次遍历都要截取字符串做hash映射
     * @param s
     * @return
     */
    public static List<String> findRepeatedDnaSequences(String s) {
        final int L = 10;
        int n = s.length();
        // 将每个字符编码成数字，10位字符刚好可以编码映射成 20 位的key
        Map<Character, Integer> codeMap = new HashMap<>();
        codeMap.put('A', 0);// 00
        codeMap.put('C', 1);// 01
        codeMap.put('G', 2);// 10
        codeMap.put('T', 3);// 11
        Map<Integer, Integer> map = new HashMap<>();
        List<String> result = new ArrayList<>();
        if (n < L) return result;
        int key = 0;
        // 初始化前9个字符构成的数字编码
        for (int i = 0; i < L - 1 ; i++) {
            key = ((key << 2) | (codeMap.get(s.charAt(i))));
        }
        // 每次取当前i数起对应第10位的下标加入key
        for (int i = 0; i <= n - L; i++) {
            // & ((1 << 20) - 1) 只保留20位，舍弃前12位
            key = ((key << 2) | (codeMap.get(s.charAt(i + L - 1)))) & ((1 << 20) - 1);
            Integer count = map.getOrDefault(key, 0);
            if (count == 1) result.add(s.substring(i, i + L));
            map.put(key, count + 1);
        }
        return result;
    }

    /**
     * 204. 计数质数
     * 埃氏筛
     * @param n
     * @return
     */
    public static int countPrimes(int n) {
        // 初始化长度为 n 的数组，将其全部置为质数
        boolean[] isPrime = new boolean[n];
        Arrays.fill(isPrime, true);
        // 埃氏筛将质数的倍数统统置为合数，遍历到 i 的平方即可
        for (int i = 2; i * i < n; i++) {
            if (isPrime[i]){
                // 这里会将范围延展至n
                for (int j = i * i; j < n; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        int count = 0;
        // 统计那些被留下来的质数
        for (int i = 2; i < isPrime.length; i++) {
            if (isPrime[i]) count++;
        }
        return count;
    }

    /**
     * 215. 数组中的第K个最大元素
     * 注意：第k大！！！！！倒着数第N-K个
     * 利用堆的思想，这里并不是建立一个nums.length的堆再排序k次，这样如果数组长度很大建堆成本很高
     * 时间复杂度 o(nlogk) 每次对堆进行调整需要执行logk的时间复杂度
     * @param nums
     * @param k
     * @return
     */
    public static int findKthLargestWithHeap(int[] nums, int k) {
        // 构建小顶堆，堆顶都是最小的，长度为k的小顶堆，堆顶放着的就是第k大的元素
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k);
        // 前k个元素入堆
        for (int i = 0; i < k; i++) {
            minHeap.offer(nums[i]);
        }
        for (int i = k; i < nums.length; i++) {
            Integer top = minHeap.peek();
            // 如果当前元素比堆顶最小元素大，那么将其加入堆中
            if (top < nums[i]) {
                minHeap.poll();
                minHeap.offer(nums[i]);
            }
        }
        return minHeap.peek();
    }

    /**
     * 215. 数组中的第K个最大元素
     * 快排的思想
     * @param nums
     * @param k
     * @return
     */
    public static int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        // 最终答案所在排序数组的位置
        int target = n - k;
        int left = 0, right = n - 1;
        while (true){
            // 进行一次划分
            int p = partition(nums, left, right);
            // 此次排序的元素刚好是目标值所在的位置的元素
            if (p == target){
                return nums[p];
                // 此次好序的元素所在位置在目标位置的左边，p 左边的元素都是小于nums[p]的，target 一定在p的右边 更新left继续排
            }else if (p < target){
                left = p + 1;
            }else {
                // p 右边的元素都是大于nums[p]的，target 一定在p的左边 更新right继续排
                right = p - 1;
            }
        }

    }

    /**
     * 划分，左右指针法
     * @param nums
     * @param left
     * @param right
     * @return
     */
    public static int partition(int[] nums, int left, int right){
        // 在区间随机选择一个元素作为标定点
        if (left < right) {
            int randomIndex = left + new Random().nextInt(right - left + 1);
            swap(nums, right, randomIndex);
        }

        // 以最后一个数字作为参照
        int pivot = nums[right];
        int begin = left, end = right;
        while (begin < end){
            // 从左向右找大于pivot的数
            while (begin < end && nums[begin] <= pivot) begin++;
            // 从右向左找小于pivot的数
            while (begin < end && nums[end] >= pivot) end--;
            if (begin < end){
                // 如果begin 和 end 尚未重复，交换他们，begin和end继续往中间查找符合条件的数据
                swap(nums, begin, end);
            }
        }
        // 最后一次指向end的值是大于pivot的，只需要让begin走到这个位置即可
        // begin 和 end 重复的位置也就是 pivot 最终应该落在的位置，交换他们
        swap(nums, begin, right);
        return begin;
    }

//    /**
//     * 220. 存在重复元素 III
//     * 滑动窗口加红黑树，红黑树插入查找效率稳定
//     * 找 [max(0, i - k), i - 1] 中最接近 nums[i] 的数
//     * @param nums
//     * @param k
//     * @param t
//     * @return
//     */
//    public static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
//        // Integer.MAX_VALUE - Integer.MIN_VALUE = 1 发生越界，需要把数字转成long进行运算
//        // 红黑树这里是大小为k的滑动窗口
//        TreeSet<Long> redBlackTree = new TreeSet<>();
//        for (int i = 0; i < nums.length; i++) {
//            long u = nums[i];
//            // 比u大的最小元素
//            Long ceiling = redBlackTree.ceiling(u);
//            // 比u小的最大元素
//            Long floor = redBlackTree.floor(u);
//            if (ceiling != null && Math.abs(ceiling - u) <= t) return true;
//            if (floor != null && Math.abs(floor - u) <= t) return true;
//            // 将当前节点加入红黑树
//            redBlackTree.add(u);
//            // 移除窗口最后一个元素
//            if (i - k >= 0){
//                redBlackTree.remove((long) nums[i - k]);
//            }
//        }
//        return false;
//    }

    /**
     * 220. 存在重复元素 III
     * 桶排序思想
     * @param nums
     * @param k
     * @param t
     * @return
     */
    public static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        // 两数之差为t的数在数轴上的间隔为t+1，只有将 size 定为 t+1，才能保证数值差为 t 的数可以落在同一个桶子里
        // 注意这里的size是指一个桶子所能容纳的元素的大小
        int size = t + 1;
        // map 存放桶下标和桶元素，不同于桶排序，这里一个桶只会出现一个元素
        Map<Long, Long> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            // 获取当前元素映射桶子的下标
            long idx = getIdx(nums[i], size);
            // 如果该元素的桶子存在且在窗口内直接可以返回，这种是存在同一个窗口刚好两个相同元素映射到同一个桶子的的情况
            if (map.containsKey(idx)) return true;
            // 找不到上诉情况下的桶子时 检查左右边桶子里有没有符合条件的数据 每个桶子内的元素值的跨度刚好为t + 1
            // 因此如果还有满足情况的数据，那么该数据一定在相隔不超过一的桶子里，也就是左右两个桶
            long left = idx - 1, right = idx + 1;
            // 检查左桶子有没有
            if (map.containsKey(left) && nums[i] - map.get(left) <= t) return true;
            // 检查右桶子有没有
            if (map.containsKey(right) && map.get(right) - nums[i] <= t) return true;
            // 没有符合条件的数据，将当前元素用来创建一个新桶
            map.put(idx, (long) nums[i]);
            // 移除掉窗口外的桶子
            if (i >= k) map.remove(getIdx(nums[i - k], size));
        }
        return false;
    }

    /**
     * 获取桶下标
     * @param num
     * @param size
     * @return
     */
    private static long getIdx(int num, int size){
        // num >= 0, num / size 表示 num 落在桶的的下标
        // num < 0, 由于 0 已经被上述条件用上了，故计算结果有误，需要将num + 1 在数轴上右移后再计算，而计算结果存在下标0
        // 但下标0的桶子已经被用掉了，故负数计算出来的桶子下标位置需要 - 1操作
        return num >= 0 ? num / size : (num + 1) / size - 1;
    }

    /**
     * 393. UTF-8 编码验证
     * @param data
     * @return
     */
    public static boolean validUtf8(int[] data) {
        int one = 1 << 7;
        int two = (1 << 7) | (1 << 6);
        for (int i = 0; i < data.length;) {
            // 对于不跟在打头元素后面 且10开头的字符都是非法字符
            if ((data[i] & two) == one) return false;
            // 单字节字符
            if ((data[i] & one) == 0){
                i++;
                continue;
            }
            // 非单字符的情况记录开头1的个数
            int count = 0;
            for (int j = 7; j >= 0; j--) {
                if (((data[i] >> j) & 1) == 1) count++;
                else break;
            }
            // utf-8最多4字符，超了就返回
            if (count > 4) return false;
            // 如果后面的元素个数不够当前多字符的长度
            if (i + count - 1 >= data.length) return false;
            for (int j = 1; j < count; j++)
                // 后续的字符不符合10开头
                if ((data[i + j] & two) != one) return false;
            i += count;
        }
        return true;
    }

    /**
     * 151. 颠倒字符串中的单词
     * @param s
     * @return
     */
    public static String reverseWords(String s) {
        String t = s.trim();
        String[] splits = t.split("\\s+");// 多个空格的正则表达式
        Collections.reverse(Arrays.asList(splits));
        StringBuilder sb = new StringBuilder();
        for (String split : splits)  sb.append(split).append(" ");
        return sb.deleteCharAt(sb.length() - 1).toString();
    }

    /**
     * 150. 逆波兰表达式求值
     * @param tokens
     * @return
     */
    public static int evalRPN(String[] tokens) {
        Deque<Integer> stack = new LinkedList<>();
        int result;
        for (String token : tokens) {
            int element1, element2;
            switch (token){
                case "+":
                    element1 = Objects.requireNonNull(stack.pollLast());
                    element2 = Objects.requireNonNull(stack.pollLast());
                    result = element2 + element1;
                    stack.addLast(result);
                    break;
                case "-":
                    element1 = Objects.requireNonNull(stack.pollLast());
                    element2 = Objects.requireNonNull(stack.pollLast());
                    result = element2 - element1;
                    stack.addLast(result);
                    break;
                case "*":
                    element1 = Objects.requireNonNull(stack.pollLast());
                    element2 = Objects.requireNonNull(stack.pollLast());
                    result = element2 * element1;
                    stack.addLast(result);
                    break;
                case "/":
                    element1 = Objects.requireNonNull(stack.pollLast());
                    element2 = Objects.requireNonNull(stack.pollLast());
                    result = element2 / element1;
                    stack.addLast(result);
                    break;
                default:
                    stack.addLast(Integer.parseInt(token));
                    break;
            }

        }
        return Objects.requireNonNull(stack.pollLast());
    }

    /**
     * 221. 最大正方形
     * @param matrix
     * @return
     */
    public static int maximalSquare(char[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dp = new int[m][n];
        int maxLen = 0;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == '1'){
                dp[i][0] = 1;
                maxLen = 1;
            }

        }
        for (int i = 0; i < n; i++) {
            if (matrix[0][i] == '1'){
                dp[0][i] = 1;
                maxLen = 1;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i-1][j-1]) + 1;
                    maxLen = Math.max(dp[i][j], maxLen);
                }
            }
        }
        return maxLen * maxLen;
    }

    /**
     * 238. 除自身以外数组的乘积
     * @param nums
     * @return
     */
    public static int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] result = new int[length];
        result[0] = 1;
        // 结果集先暂存元素下标对应左半部分的数据的乘积
        for (int i = 1; i < length; i++) {
            result[i] = nums[i - 1] * result[i - 1];
        }
        // 声明一个临时变量用于记录元素i对应的右半部分乘积
        int temp = 1;
        for (int i = length - 1; i >= 0; i--) {
            result[i] = result[i] * temp;
            // i 走完了更新 下标i-1对应右半部分元素乘积为 temp *= nums[i];
            temp *= nums[i];
        }
        return result;
    }

    /**
     * 400. 第 N 位数字
     * todo
     * @param n
     * @return
     */
    public static int findNthDigit(int n) {
        // d 位数有 9*(10^d-1)个，d 位数的所有位数有 d*9*(10^d-1)个
        int d = 1, count = 9;
        double sumD;
        // 找到n所在的数字位数d
        while ((sumD = d * count * Math.pow(10, d - 1)) <= n){
            n -= sumD;
            d++;
        }
        int num = n / d;// 确定是第几个数

        return 0;
    }


    /**
     * 279. 完全平方数
     * 动态规划
     * @param n
     * @return
     */
    public static int numSquares(int n) {
        int[] dp = new int[n + 1];
        for (int i = 0; i < dp.length; i++) {
            dp[i] = i;
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

}
