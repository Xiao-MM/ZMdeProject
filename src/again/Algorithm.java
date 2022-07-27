package again;

import java.util.HashMap;
import java.util.Map;

public class Algorithm {

    public static void main(String[] args) {
        System.out.println(lengthOfLongestSubstring("aabcdaacbbadd"));
    }

    /**
     * 3. 无重复字符的最长子串
     * todo 重做
     * abcabcbb
     * @param s
     * @return
     */
    public static int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) return 0;
        // k = 出现重复的字符， v = 左指针需要下移的下一个位置
        Map<Character, Integer> map = new HashMap<>();
        // l 记录窗口子串的左边界下标位置，r 当前窗口的右边界
        int l = 0, maxLen = 0;
        for (int r = 0; r < s.length(); r++) {
            // 窗口串中出现了重复字符，更新左边界为重复元素的下一个位置
            if (map.containsKey(s.charAt(r))){
                l = Math.max(l, map.get(s.charAt(r)));
            }
            // 将
            map.put(s.charAt(r), r + 1);
            maxLen = Math.max(maxLen, r - l + 1);
        }
        return maxLen;
    }
}
