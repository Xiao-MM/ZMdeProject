package algorithm;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * 组合总数
 */
public class CombinedSum {

    List<List<Integer>> results = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        backtrack(candidates,target,0,0, path);
        return results;
    }

    private void backtrack(int[] candidates, int target, int startIndex,  int sum, LinkedList<Integer> path){
        // 结束条件
        if (sum == target){
            results.add(new ArrayList<>(path));
            return;
        }
        for (int i = startIndex; i < candidates.length; i++) {
            if (sum + candidates[i] > target) {// 此处不能用sum > target来判断剪枝
                continue;
            }
            path.add(candidates[i]);
            backtrack(candidates, target, i, sum + candidates[i], path);
            path.removeLast();
        }
    }

}
