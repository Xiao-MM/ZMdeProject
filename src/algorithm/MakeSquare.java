package algorithm;


import java.util.*;

public class MakeSquare {

    /**
     * 473. 火柴拼正方形
     * @param matchsticks
     * @return
     */
    public static boolean makesquare(int[] matchsticks) {
        int sum = 0;
        for (int matchstick : matchsticks) sum += matchstick;
        // 总长度不是4的整数倍返回false
        if (sum % 4 != 0) return false;
        int perLen = sum / 4;
        // 每根的长度不能超过平均长度
        for (int matchstick : matchsticks) {
            if (perLen < matchstick) return false;
        }
        Arrays.sort(matchsticks);
        // 四条边已被火柴拼接好的长度
        int[] edges = new int[4];
        // 标记数组，标记火柴是否已被使用
        boolean[] used = new boolean[matchsticks.length];
        // 递归判断
        return dfs(matchsticks, matchsticks.length - 1, edges, 0, used, perLen);
    }

    /**
     *
     * @param matchsticks 火柴数组
     * @param end 火柴下标
     * @param edges 边数组，值为已放置的长度
     * @param k 边数组下标
     * @param used 火柴i是否已经使用
     * @param perLen 每条边的长度
     * @return
     */
    private static boolean dfs(int[] matchsticks, int end, int[] edges, int k, boolean[] used, int perLen){
        // 正方形的4条边都放完了可以组成正方形
        if (k == edges.length) return true;
        // 如果第k条边已经放满了，则继续方k+1条边
        if (edges[k] == perLen){
            return dfs(matchsticks, matchsticks.length - 1, edges, k + 1, used, perLen);
        }
        for (int i = end; i >= 0; i--) {
            // 第i个已经被使用
            if (used[i]) continue;
            // 第i个火柴不可用
            if (edges[k] + matchsticks[i] > perLen) continue;

            // 将数量加入当前边长
            edges[k] += matchsticks[i];
            used[i] = true;
            // 继续向前检索
            if (dfs(matchsticks, i - 1, edges, k, used, perLen)) return true;
            // 走不通就回退
            edges[k] -= matchsticks[i];
            used[i] = false;

            // 如果出现重复数据，则不行都不行，找到下一个不重复的火柴进行尝试
            while (i > 0 && matchsticks[i] == matchsticks[i - 1]){
                i--;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        System.out.println(makesquare(new int[]{1,1,1,1,1,2,2,2,3,3,3}));
    }
}
