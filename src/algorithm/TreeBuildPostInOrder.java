package algorithm;

import java.util.HashMap;
import java.util.Map;

/**
 * 106. 从中序与后序遍历序列构造二叉树
 */
public class TreeBuildPostInOrder{
    Map<Integer,Integer> inorderMap;
    int[] postorder;

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        this.postorder = postorder;
        inorderMap = new HashMap<>();
        int length = postorder.length;
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i],i);
        }
        return build(length - 1, 0, length - 1);
    }

    private TreeNode build(int rootIdx, int leftIdx, int rightIdx){
        if (leftIdx > rightIdx) return null;
        TreeNode node = new TreeNode(postorder[rootIdx]);
        Integer i = inorderMap.get(postorder[rootIdx]);// 获取后序遍历的根节点在中序遍历中的位置
        node.right = build(rootIdx - 1, i + 1, rightIdx);
        node.left = build(rootIdx - (rightIdx - i) - 1, leftIdx, i - 1);// 左子树的根节点在 rootIdx - 右子树长度 - 1 的位置上
        return node;
    }

    public static void main(String[] args) {
        TreeBuildPostInOrder treeBuildPostInOrder = new TreeBuildPostInOrder();
        TreeNode root = treeBuildPostInOrder.buildTree(new int[]{3,9,20,15,7}, new int[]{9,3,15,20,7});
        TreeNode.printTree(root);
    }
}
