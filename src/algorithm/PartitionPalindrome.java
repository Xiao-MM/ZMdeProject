package algorithm;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * 131. 分割回文串
 */
public class PartitionPalindrome {

    List<List<String>> result = new ArrayList<>();
    LinkedList<String> path = new LinkedList<>();
    // 动态规划的字符串状态数组
    boolean[][] dp;

    public List<List<String>> partition(String s) {
        if (s == null || s.length() == 0) return result;
        int length = s.length();
        dp = new boolean[length][length];
        char[] chars = s.toCharArray();

        // 预处理s，dp记录着所有情况的子串信息
        for (int right = 0; right < length; right++) {
            for (int left = 0; left <= right; left++) {
                // s[i]==s[j] && 子串长度为小于3的都是回文串
                if (chars[left] == chars[right] && (right - left <= 2 || dp[left + 1][right - 1]))
                    dp[left][right] = true;
            }
        }
        dfs(s, 0, length);
        return result;
    }

    /**
     * 递归回溯所有的可能
     * @param s
     * @param start
     * @param length
     */
    private void dfs(String s, int start, int length){
        if (start == length){
            result.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i < length; i++) {
            // 剪枝，如果前缀是回文串则进行后续操作
            if (dp[start][i]){
                path.addLast(s.substring(start, i + 1));
                dfs(s, i + 1, length);
                path.removeLast();
            }
        }
    }

    public static void main(String[] args) {
        PartitionPalindrome partitionPalindrome = new PartitionPalindrome();
        System.out.println(partitionPalindrome.partition("aaba"));
    }
}
