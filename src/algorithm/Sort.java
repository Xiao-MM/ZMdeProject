package algorithm;

import java.util.Arrays;
import java.util.Random;

public class Sort {

    /**
     * 快速排序
     * @param nums
     * @param left
     * @param right
     */
    public static void quickSort(int[] nums, int left, int right){
        if (left < right){
            int partition = partition(nums, left, right);
            quickSort(nums, left, partition - 1);
            quickSort(nums, partition + 1, right);
        }
    }



    /**
     * 划分，左右指针法
     * @param nums
     * @param left
     * @param right
     * @return
     */
    public static int partition(int[] nums, int left, int right){
        // 在区间随机选择一个元素作为标定点
        Random random = new Random();
        if (left < right) {
            int randomIndex = left + random.nextInt(right - left + 1);
            swap(nums, right, randomIndex);
        }
        // 以最后一个数字作为参照
        int pivot = nums[right];
        int begin = left, end = right;
        while (begin < end){
            // 从左向右找大于pivot的数
            while (begin < end && nums[begin] <= pivot) begin++;
            // 从右向左找小于pivot的数
            while (begin < end && nums[end] >= pivot) end--;
            if (begin < end){
                // 如果begin 和 end 尚未重复，交换他们，begin和end继续往中间查找符合条件的数据
                swap(nums, begin, end);
            }
        }
        // 最后一次指向end的值是大于pivot的，只需要让begin走到这个位置即可
        // begin 和 end 重复的位置也就是 pivot 最终应该落在的位置，交换他们
        swap(nums, begin, right);
        return begin;
    }

    public static void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public static void main(String[] args) {
        int[] nums = new int[]{2,3,1,4,5,7,6};
        quickSort(nums, 0, nums.length - 1);
        System.out.println(Arrays.toString(nums));

    }
}
