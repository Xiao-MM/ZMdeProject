package algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * 40. 组合总和 II
 */
public class CombinedSum2 {

    List<List<Integer>> result = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates.length == 0){
            return result;
        }
        // 排序用于去重同层剪枝
        Arrays.sort(candidates);
        backtrack(candidates, target, 0);
        return result;
    }

    /**
     * dfs
     * @param candidates 原数组
     * @param target 目标值，每一次遍历减去上一层方法的值
     * @param start 起始循环位置
     */
    void backtrack(int[] candidates, int target, int start){
        if (target == 0){
            result.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i < candidates.length; i++) {
            // 大剪枝，减去和会大于target的后续分支
            if (target - candidates[i] < 0){
                return;
            }
            // 小剪枝，减去同一层的相同元素
            if (i > start && candidates[i-1] == candidates[i]){
                continue;
            }
            path.add(candidates[i]);
            backtrack(candidates, target - candidates[i], i + 1);
            path.removeLast();
        }
    }

}
