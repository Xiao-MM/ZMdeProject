
import java.util.*;

public class StaticMethod {
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
     * 翻转整数
     * digit = x%10
     * num = (x/10)*10 + digit
     * @param x 带翻转数
     * @return
     */
    public static int reverse(int x){
        int digit = 0;
        int rev = 0;
        while (x != 0){
            if (rev < Integer.MIN_VALUE/10 || rev > Integer.MAX_VALUE/10){
                return 0;
            }
            digit = x%10;
            x/=10;
            rev = rev*10 + digit;
        }
        return rev;
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
     * 动态规划思想
     * 状态转移方程 p(i,j) = p(i+1,j-1)&&(s[i]==s[j])
     * @param s j-1 - (i+1) + 1 = j-i-1
     * @return
     */
    public static String longestPalindrome(String s) {
        char[] chars = s.toCharArray();
        if (s.length() == 0 || s.length() == 1) return s;
        if (s.length() == 2 && chars[0] == chars[1]) return s;
        // 状态字数组
        boolean[][] status = new boolean[s.length()][s.length()];
        for (int i = 0; i < s.length(); i++) {
            status[i][i] = true;
        }
        int begin = 0;
        int maxLen = 1;
        for (int j = 1; j < s.length() - 1;j++) {
            for (int i = j+1; j < j-i-1; j++) {
                if (status[i+1][j-1] && chars[i]==chars[j]){
                    status[i][j] = true;
                    begin = i;
                    maxLen = j - i + 1;
                }else {
                    status[i][j] = false;
                }
            }
        }
        return s.substring(begin,maxLen);
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
     * 三数之和 双指针
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
            if(nums[i] > 0) break; // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
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
        int[] dp = new int[triangle.size() + 1]; // 多申请一个元素位置避免数组下标越界
        for (int i = triangle.size() - 1; i >= 0 ; i--) {
            for (int j = 0; j <= i; j++) {
               dp[j] = Math.min(dp[j],dp[j+1]) + triangle.get(i).get(j);// dp[3] = min(dp[3],dp[4]) + triangle[3][j] = min(0,0) +
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
}
