package entity;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * 78. 子集
 * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
 *
 * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
 */
public class Subsets {
    List<List<Integer>> result = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0){
            result.add(new ArrayList<>());
            return result;
        }
        backtrack(nums, 0, path);
        return result;
    }

    private void backtrack(int[] nums, int start, LinkedList<Integer> path){
        // 没有结束条件
        result.add(new ArrayList<>(path));
        for (int i = start; i < nums.length; i++) {// for循环结束时递归也就结束了
            path.add(nums[i]);
            backtrack(nums,i + 1, path);
            path.removeLast();
        }
    }
}
