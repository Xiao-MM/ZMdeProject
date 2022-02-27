package algorithm;


import java.util.*;

public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    /**
     * 102. 二叉树的层序遍历
     * @param root
     * @return
     */
    public static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                level.add(node.val);
                if (node.left != null)
                    queue.offer(node.left);
                if (node.right != null)
                    queue.offer(node.right);
            }
            result.add(level);
        }
        return result;
    }

    /**
     * 103. 二叉树的锯齿形层序遍历
     * @param root
     * @return
     */
    public static List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean isReverse = false;
        while (!queue.isEmpty()){
            int size = queue.size();
            LinkedList<Integer> level = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (isReverse){
                    level.addFirst(node.val);
                }else {
                    level.addLast(node.val);
                }
                if (node.left != null)
                    queue.offer(node.left);
                if (node.right != null)
                    queue.offer(node.right);
            }
            isReverse = !isReverse;
            result.add(level);
        }
        return result;
    }

    /**
     * 101. 对称二叉树
     * @param root
     * @return
     */
    public static boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return dfs(root.left, root.right);
    }

    private static boolean dfs(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        if (left.val != right.val) return false;
        return dfs(left.left, right.right) && dfs(left.right, right.left);
    }

    /**
     * 112. 路径总和
     * @param root
     * @param targetSum
     * @return
     */
    public static boolean hasPathSum(TreeNode root, int targetSum) {
        // 如果直接走到空节点直接证明路径不存在
        if (root == null){
            return false;
        }
        // 如果走到叶子节点且targetSum刚好减到和叶子节点值相同时证明路径存在
        if (root.left == null && root.right == null){
            return targetSum == root.val;
        }
        // 递归判断左右子树
        return hasPathSum(root.left, targetSum - root.val)
                || hasPathSum(root.right, targetSum - root.val);
    }

    List<List<Integer>> result;

    LinkedList<Integer> path;

    /**
     * 113. 路径总和 II
     * @param root
     * @param targetSum
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        this.result = new ArrayList<>();
        this.path = new LinkedList<>();
        pathSumDFS(root, targetSum);
        return result;
    }

    private void pathSumDFS(TreeNode root, int targetSum){
        // 节点为空直接返回
        if (root == null) return;
        // 将当前节点加入路径
        path.addLast(root.val);
        // 找到符合要求的路径添加
        if (root.left == null && root.right == null && targetSum == root.val){
            result.add(new ArrayList<>(path));// 注意这里不能return，return后面的撤销命令不会执行
        }
        // 在左子树中找
        pathSumDFS(root.left, targetSum - root.val);
        // 在右子树中找
        pathSumDFS(root.right, targetSum - root.val);
        // 回退时移除路径的节点数据
        path.removeLast();
    }

    /**
     * 107. 二叉树的层序遍历 II
     * @param root
     * @return
     */
    public static List<List<Integer>> levelOrderBottom(TreeNode root) {
        LinkedList<List<Integer>> result = new LinkedList<>();
        LinkedList<TreeNode> queue = new LinkedList<>();
        if (root == null) return result;
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            List<Integer> path = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                path.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            result.addFirst(path);
        }
        return result;
    }

//    /**
//     * 110. 平衡二叉树 效率低
//     * @param root
//     * @return
//     */
//    public static boolean isBalanced(TreeNode root) {
//       if (root == null) return true;
//       return Math.abs(maxDepth(root.left) - maxDepth(root.right)) <= 1
//               && isBalanced(root.left) && isBalanced(root.right);
//    }

    /**
     * 110. 平衡二叉树
     * @param root
     * @return
     */
    public static boolean isBalanced(TreeNode root) {
        return recur(root) != -1;
    }

    private static int recur(TreeNode root){
        // 当节点为空节点返回树高0
        if (root == null) return 0;
        int left = recur(root.left);
        // 左子树非平衡，剪枝
        if (left == -1) return -1;
        int right = recur(root.right);
        // 右子树非平衡，剪枝
        if (right == -1) return -1;
        // 判断左右子树高度差是否满足平衡条件
        return Math.abs(left - right) <= 1 ? Math.max(left, right) + 1 : -1;
    }


    /**
     * 104. 二叉树的最大深度
     * @param root
     * @return
     */
    public static int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }



    /**
     * 114. 二叉树展开为链表
     * @param root
     */
    public static void flatten(TreeNode root) {

    }

    public static void printTree(TreeNode root){
        if (root != null){
            System.out.print(root.val + " ");
            printTree(root.left);
            printTree(root.right);
        }
    }

    public static void main(String[] args) {
//        TreeNode root = new TreeNode(3,
//                new TreeNode(9),
//                new TreeNode(20,
//                        new TreeNode(15),
//                        new TreeNode(7)));
        TreeNode root = new TreeNode(3, new TreeNode(1), new TreeNode(9, null, new TreeNode(2)));

//        System.out.println(zigzagLevelOrder(root));
//        System.out.println(maxDepth(root));
//        System.out.println(hasPathSum(root, 19));
//        System.out.println(root.pathSum(root, 19));
//        System.out.println(levelOrderBottom(root));
        System.out.println(isBalanced(root));
    }



}

