package entity;

import java.util.*;

/**
 * 全排列
 */
public class Permutations {

    List<List<Integer>> results = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();
    List<String> strResult = new ArrayList<>();
    StringBuilder charPath = new StringBuilder();
    boolean[] used;

    /**
     * 无重复数据的全排列
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        used = new boolean[nums.length];
        backtrack(nums,used,path);
        return results;
    }

    private void backtrack(int[] nums, boolean[] used, LinkedList<Integer> path){
        if (path.size() == nums.length){
            results.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]){ // i 已经被使用过了，走下一个
                continue;
            }
            path.add(nums[i]);
            used[i] = true;
            backtrack(nums, used, path);
            used[i] = false;
            path.removeLast();
        }
    }


    /**
     * 含重复数据的全排列
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        // 需要先将nums排序，让重复元素紧密贴在一起方便剪枝
        Arrays.sort(nums);
        used = new boolean[nums.length];
        backtrackUnique(nums,used,path);
        return results;
    }

    private void backtrackUnique(int[] nums, boolean[] used, LinkedList<Integer> path){
        if (path.size() == nums.length){
            results.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!used[i]){
                // 剪枝 如果当前数据不是第一个元素且和上一个元素相同并且上一个已经不可用之后跳过该步骤
                if (i > 0 && nums[i] == nums[i-1] && !used[i-1]){
                    continue;
                }
                path.add(nums[i]);
                used[i] = true;
                backtrackUnique(nums, used, path);
                used[i] = false;
                path.removeLast();
            }
        }
    }

    public String[] permutation(String s) {
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        s = new String(chars);
        used = new boolean[chars.length];
        backtrackUnique(s, used, charPath);
        String[] strings = new String[strResult.size()];
        for (int i = 0; i < strResult.size(); i++) {
            strings[i] = strResult.get(i);
        }
        return strings;
    }

    private void backtrackUnique(String s, boolean[] used, StringBuilder path){
        if (path.length() == s.length()){
            strResult.add(path.toString());
            return;
        }
        for (int i = 0; i < s.length(); i++) {
            if (!used[i]){
                // 剪枝 如果当前数据不是第一个元素且和上一个元素相同并且上一个已经不可用之后跳过该步骤
                if (i > 0 && s.charAt(i) ==  s.charAt(i-1) && !used[i-1]){
                    continue;
                }
                path.append(s.charAt(i));
                used[i] = true;
                backtrackUnique(s, used, path);
                used[i] = false;
                path.deleteCharAt(path.length()-1);// 注意这里不是移除i，而是移除path的最后一个元素
            }
        }
    }
}
