package algorithm;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * 组合
 */
public class Combination {

    List<List<Integer>> result = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();
    
    public List<List<Integer>> combine(int n, int k) {
        backtrack(n, k,1, path);
        return result;
    }

    private void backtrack(int n, int k, int start, LinkedList<Integer> path){
        if (path.size() == k){ // 当path长度达到所需的长度即可返回
            result.add(new ArrayList<>(path));
            return;
        }
        // 考虑剪枝 搜索起点的上界 = n - (k - path.size()) + 1
        // for (int i = start; i <= n; i++) {
        for (int i = start; i <= n - (k - path.size()) + 1; i++) {
            path.add(i);
            backtrack(n, k, i+1, path);
            path.removeLast();
        }
    }
}
