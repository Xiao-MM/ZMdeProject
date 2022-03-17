package algorithm;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class CombinationSum3 {

    LinkedList<Integer> path = new LinkedList<>();
    List<List<Integer>> result = new ArrayList<>();
    /**
     * 216. 组合总和 III
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSum3(int k, int n) {
        dfs(n, 1, k);
        return result;
    }

    private void dfs(int sum, int begin, int k){
        // sum < 0 直接返回
        if (sum < 0) return;
        // 超过个数没有找到直接返回
        if (path.size() > k) return;
        // k个数且和为n加入结果集
        if (path.size() == k && sum == 0){
            result.add(new ArrayList<>(path));
            return;
        }
        // 依次从当前位置到9找起
        for (int i = begin; i <= 9; i++) {
            path.addLast(i);
            // 带着减过的值去找下一个数
            dfs(sum - i, i + 1, k);
            path.removeLast();
        }
    }

    public static void main(String[] args) {
        CombinationSum3 sum3 = new CombinationSum3();
        System.out.println(sum3.combinationSum3(3, 7));
    }
}
