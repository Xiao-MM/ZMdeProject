

import entity.Combination;
import entity.CombinedSum;
import entity.Permutations;

import java.util.*;

public class Main {
    public static void main(String[] args) {
//        int x = 123000;
//        int n = StaticMethod.getNumLength(x);
//        System.out.println(x + "的位数是：" + n);
//        int reverse = StaticMethod.reverse(x);
//        System.out.println(x + "反转后：" + reverse);
//        int[] nums = new int[]{2,3,5,6};
//        int target = 9;
//        System.out.println(Arrays.toString(twoSum(nums, target)));
//        int[] nums = new int[]{7,1,2,5,4,3,6};
//        int[] nums = new int[]{0,1,0,3,2,3};
//        int[] nums = new int[]{1,3,6,7,9,4,10,5,6};
//        int[] nums = new int[]{10,9,2,5,3,7,101,18};
//        System.out.println(StaticMethod.officialLengthOfLIS(nums));
//        System.out.println(StaticMethod.fastMulti(2,10));
//        System.out.println(StaticMethod.fib_matrix(5));
        //System.out.println(StaticMethod.isPalindrome("aa"));
//        System.out.println(StaticMethod.longestPalindrome("aabaa"));

//        StaticMethod staticMethod = new StaticMethod();
//        List<List<Integer>> combine = staticMethod.combine(10, 3);
//        System.out.println(combine);
//        int[] nums = new int[]{1,2,3};
//        Permutations permutations = new Permutations();
//        String s = "aca";
//        String[] strPermutations = permutations.permutation(s);
//        System.out.println(Arrays.toString(strPermutations));
//        List<List<Integer>> permute = permutations.permute(nums);
//        System.out.println(permute);
//        int[] nums1 = new int[]{1,3,1};
//        List<List<Integer>> permuteUnique = permutations.permuteUnique(nums1);
//        System.out.println(permuteUnique);
//        int[] candidates = new int[]{2,3,6,7};
//        int target = 7;
//        CombinedSum combinedSum = new CombinedSum();
//        List<List<Integer>> combinations = combinedSum.combinationSum(candidates, target);
//        System.out.println(combinations);
//        Combination combination =new Combination();
//        List<List<Integer>> combine = combination.combine(4, 2);
//        System.out.println(combine);
//        int length = StaticMethod.lengthOfLongestSubstring("abcdabcbb");
//        System.out.println(length);
//        int[] nums = new int[]{-2,0,0,2,2};
//        List<List<Integer>> result = StaticMethod.threeSum(nums);
//        System.out.println(result);
        List<Integer> row1 = new ArrayList<>(Collections.singletonList(2));
        List<Integer> row2 = new ArrayList<>(Arrays.asList(3,4));
        List<Integer> row3 = new ArrayList<>(Arrays.asList(6,5,7));
        List<Integer> row4 = new ArrayList<>(Arrays.asList(4,1,8,3));
        List<List<Integer>> triangle = new ArrayList<>(Arrays.asList(row1,row2,row3,row4));
        int i = StaticMethod.minimumTotal(triangle);
        System.out.println(i);
    }
}
