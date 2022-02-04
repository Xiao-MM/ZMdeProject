package algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * 90. 子集 II
 */
public class Subsets2 {
    List<List<Integer>> result = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        // 排序方便去重
        Arrays.sort(nums);
        backtrack(nums,0);
        return result;
    }

    /**
     * 深度优先遍历
     * @param nums 数组
     * @param start 深度遍历的下一个起始位置
     */
    void backtrack(int[] nums, int start){
        result.add(new ArrayList<>(path));
        for (int i = start; i < nums.length; i++) {
            // 如果当前元素和上一个元素相同，证明该元素已经被包含了，再遍历会发生重复，直接剪枝走下一个元素
            if (i > start && nums[i] == nums[i-1]){
                continue;
            }
            path.add(nums[i]);
            // 每次递归的位置+1
            backtrack(nums, i + 1);
            path.removeLast();
        }
    }

}
