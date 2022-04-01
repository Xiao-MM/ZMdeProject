package algorithm;


import java.util.Arrays;

/**
 * 归并排序
 */
public class MergeSort {

    public static void mergeSort(int[] nums) {
        int[] temp = new int[nums.length];
        sort(nums, 0, nums.length - 1, temp);
    }

    /**
     * 分治合并
     * @param nums
     * @param left
     * @param right
     * @param temp
     */
    public static void sort(int[] nums, int left, int right, int[] temp){
        if (left < right){
            // 选取中点
            int mid = (left + right) >> 1;
            // 划分左半区域
            sort(nums, left, mid, temp);
            // 划分右半区域
            sort(nums, mid + 1, right, temp);
            // 合并左半区域和右半区域
            merge(nums, left, mid, right, temp);
        }
    }

    /**
     * 归并数组
     * @param nums
     * @param left
     * @param mid
     * @param right
     * @param temp
     */
    public static void merge(int[] nums, int left, int mid, int right, int[] temp){
        int i = left, j = mid + 1, t = 0;
        while (i <= mid && j <= right){
            if (nums[i] < nums[j])
                temp[t++] = nums[i++];
            else
                temp[t++] = nums[j++];
        }
        // 左半区域还有剩余，直接加到结尾
        while (i <= mid) temp[t++] = nums[i++];
        // 右半区域还有剩余，直接加到结尾
        while (j <= right) temp[t++] = nums[j++];
        t = 0;
        // 将归并后的数组合并到原数组
        while (left <= right) nums[left++] = temp[t++];
    }

    public static void main(String[] args) {
        int[] nums = new int[]{2,3,5,1,8,4,6};
        mergeSort(nums);
        System.out.println(Arrays.toString(nums));
    }
}
