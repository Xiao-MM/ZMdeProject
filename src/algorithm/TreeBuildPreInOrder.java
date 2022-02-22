package algorithm;

import java.util.HashMap;
import java.util.Map;

/**
 * 105. 从前序与中序遍历序列构造二叉树
 * pre_root_idx + (idx-1 - in_left_idx +1)  + 1
 * preorder: [3,9,20,15,7] 3-0 9-1 "20-2" root + (i - 1 - 0 + 1) + 1 = 0 + (1-1-0+1) +1 = 2
 * inorder: [9,3,15,20,7] 3-1 9-0
 */
public class TreeBuildPreInOrder {
    int[] preorder;
    Map<Integer,Integer> inorderMap = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i],i);
        }
        return build(0, 0, preorder.length - 1);
    }

    /**
     * 以前序遍历下标为基准，根据中序遍历去划分前序的区间
     * @param root 根节点在前序数组的下标 先序遍历的索引
     * @param left 要进行划分的区间范围左边界 中序遍历的索引
     * @param right 要进行划分的区间范围右边界 中序遍历的索引
     * @return 根节点
     */
    TreeNode build(int root, int left, int right){
        if (left > right) return null;// 当left > right时发生越界，结束程序
        TreeNode node = new TreeNode(preorder[root]);
        Integer idx = inorderMap.get(preorder[root]);// 获取根在中序的位置
        node.left = build(root + 1, left, idx - 1);// 划分前序遍历的左区间
        // 右子树的根的索引为先序中的 当前根位置 + 左子树的数量 + 1 = root + (idx - 1 - left + 1) + 1
        node.right = build(root + idx - left + 1,idx + 1, right);// 划分前序遍历的右区间
        return node;
    }

    public static void main(String[] args) {
        TreeBuildPreInOrder treeBuildPreInOrder = new TreeBuildPreInOrder();
        TreeNode root = treeBuildPreInOrder.buildTree(new int[]{3,9,20,15,7}, new int[]{9,3,15,20,7});
        TreeNode.printTree(root);
    }
}
