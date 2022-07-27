package algorithm;


import java.util.*;

@SuppressWarnings("all")
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

    static class Node {
        public int val;
        public List<Node> children;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }

        /**
         * 429. N 叉树的层序遍历
         * @param root
         * @return
         */
        public List<List<Integer>> levelOrder(Node root) {
            List<List<Integer>> result = new ArrayList<>();
            if (root == null) return result;
            Queue<Node> q = new LinkedList<>();
            q.offer(root);
            while (!q.isEmpty()){
                int size = q.size();
                List<Integer> path = new ArrayList<>();
                for (int i = 0; i < size; i++) {
                    Node node = q.poll();
                    path.add(node.val);
                    if (node.children != null && node.children.size() > 0){
                        for (Node child : node.children) {
                            q.offer(child);
                        }
                    }
                }
                result.add(path);
            }
            return result;
        }
    };

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

    static class Symmetric{
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


    static class pathSum{

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

    /**
     * 199. 二叉树的右视图
     * BFS
     * @param root
     * @return
     */
    public static List<Integer> rightSideViewBFS(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (i == size - 1) result.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
        }
        return result;
    }


    static class RightSideView{
        List<Integer> result = new ArrayList<>();
        /**
         * 199. 二叉树的右视图
         * DFS
         * @param root
         * @return
         */
        public List<Integer> rightSideView(TreeNode root) {
            rightSideDFS(root, 0);
            return result;
        }

        /**
         * 根右左的顺序遍历
         * @param root
         * @param depth
         */
        private void rightSideDFS(TreeNode root, int depth){
            if (root == null) return;
            // 当高度满足时即是所需数据
            if (result.size() == depth)
                result.add(root.val);
            depth++;
            rightSideDFS(root.right, depth);
            rightSideDFS(root.left, depth);
        }

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
        while (root != null){
            if (root.left != null) {
                TreeNode pre = root.left;
                while (pre.right != null)
                    pre = pre.right;
                // 将root的右子树接到pre右下方
                pre.right = root.right;
                // 将root的左子树接到右子树上
                root.right = root.left;
                root.left = null;
            }
            root = root.right;
        }
    }

    public static void printTree(TreeNode root){
        if (root != null){
            System.out.print(root.val + " ");
            printTree(root.left);
            printTree(root.right);
        }
    }

    static class SortedListToBST{
        ListNode globalNode;

        /**
         * 109. 有序链表转换二叉搜索树
         * @param head
         * @return
         */
        public TreeNode sortedListToBST(ListNode head) {
            globalNode = head;
            int length = getLength(head);
            return buildTree(0, length - 1);
        }

        /**
         * 获取链表长度
         * @param head
         * @return
         */
        private int getLength(ListNode head){
            int length = 0;
            while (head != null){
                length++;
                head = head.next;
            }
            return length;
        }

        /**
         * 链表采用二分构建根节点
         * @param left 左边界
         * @param right 右边界
         * @return
         */
        private TreeNode buildTree(int left, int right){
            // 如果左边界 > 右边界 构建空节点
            if (left > right) return null;
            // 根据left right 二分划分区间
            int mid = (left + right) >> 1;
            // 创建根节点，先占个坑，当下一条执行完再填充数据
            TreeNode node = new TreeNode();
            // 递归建立左子树
            node.left = buildTree(left, mid - 1);
            // 中序遍历填充节点值
            node.val = globalNode.val;
            // 链表顺序访问
            globalNode = globalNode.next;
            // 递归建立右子树
            node.right = buildTree(mid + 1, right);
            return node;
        }
    }


    static class ValidBST{
        /**
         * 全局变量记录前驱节点的值
         */
        long pre = Long.MIN_VALUE;

        /**
         * 98. 验证二叉搜索树
         * @param root
         * @return
         */
        public boolean isValidBST(TreeNode root) {
            // 空节点是二叉搜索树
            if (root == null) return true;
            // 如果左子树不是二叉搜索树返回false
            if (!isValidBST(root.left)) return false;
            // 前驱节点大于当前节点返回false
            if (root.val <= pre) return false;
            // 更新前驱
            pre = root.val;
            // 去判断右子树是不是二叉搜索树
            return isValidBST(root.right);
        }
    }


    static class KthSmallest{
        int k, ans;
        /**
         * 230. 二叉搜索树中第K小的元素
         * @param root
         * @param k
         * @return
         */
        public int kthSmallest(TreeNode root, int k) {
            this.k = k;
            dfs(root);
            return ans;
        }

        void dfs(TreeNode root){
            if (root == null) return;
            dfs(root.left);
            if (--k == 0) ans = root.val;
            dfs(root.right);
        }
    }

    /**
     * 94. 二叉树的中序遍历
     * 迭代算法
     * @param root
     * @return
     */
    public static List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Deque<TreeNode> stack = new LinkedList<>();
        while (root != null || !stack.isEmpty()){
            // 把所有左边的节点入栈
            while (root != null){
                stack.push(root);
                root = root.left;
            }
            // 当左子树为空的时候弹出
            TreeNode pop = stack.pop();
            result.add(pop.val);
            // 然后把右子树作为root
            root = pop.right;
        }
        return result;
    }

    static class PostorderTraversal{
        /**
         * 145. 二叉树的后序遍历
         * @param root
         * @return
         */
        public static List<Integer> postorderTraversal(TreeNode root) {
            List<Integer> result = new ArrayList<>();
            dfs(root, result);
            return result;
        }

        private static void dfs(TreeNode root, List<Integer> result){
            if (root == null) return;
            dfs(root.left, result);
            dfs(root.right, result);
            result.add(root.val);
        }
    }

    static class RecoverTree{

        TreeNode preNode = new TreeNode(Integer.MIN_VALUE);
        TreeNode firstNode = null;
        TreeNode secondNode = null;

        /**
         * 99. 恢复二叉搜索树
         * @param root
         */
        public void recoverTree(TreeNode root) {
            inOrder(root);
            int temp = firstNode.val;
            firstNode.val = secondNode.val;
            secondNode.val = temp;
        }

        /**
         * 中序遍历查找待交换的节点
         * @param root
         */
        private void inOrder(TreeNode root){
            if (root == null) return;
            inOrder(root.left);
            if (preNode.val > root.val){
                // 注意这里不是if else 的关系
                if (firstNode == null) firstNode = preNode;
                // 如果上一行执行了这一行也跟着执行，只不过随着递归继续进行下去secondNode可能会被新的值取代掉
                if (firstNode != null) secondNode = root;
            }
            preNode = root;
            inOrder(root.right);
        }
    }

    static class SumNumbers{
        /**
         * 129. 求根节点到叶节点数字之和
         * @param root
         * @return
         */
        public int sumNumbers(TreeNode root) {
            return dfs(root, 0);
        }

        private int dfs(TreeNode root, int product){
            // 主要针对根节点为空的情况
            if (root == null) return 0;
            // 从根节点到该层节点的数值
            product = product * 10 + root.val;
            // 到达叶子节点直接return到该叶子节点的路径表示数字
            if (root.left == null && root.right == null) return product;
            // 累加每个分支的数值
            return dfs(root.left, product) + dfs(root.right, product);
        }
    }

    /**
     * 236. 二叉树的最近公共祖先
     * @param root
     * @param p
     * @param q
     * @return
     */
    public static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 节点为空返回空
        if (root == null) return null;
        // 节点为p,q直接返回root
        if (root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        // 孩子落在两边root即为祖先
        if (left != null && right != null) return root;
        // 左孩子有右孩子没有，直接返回左孩子
        if (left != null && right == null) return left;
        // 右孩子有左孩子没有，直接返回右孩子
        if (left == null && right != null) return right;
        // 都没有直接返回null
        return null;
    }




    public static void main(String[] args) {
//        TreeNode root = new TreeNode(3,
//                new TreeNode(1), new TreeNode(9,
//                                            new TreeNode(4), null ));
//        System.out.println(root.isValidBST(root));

//        System.out.println(zigzagLevelOrder(root));
//        System.out.println(maxDepth(root));
//        System.out.println(hasPathSum(root, 19));
//        System.out.println(root.pathSum(root, 19));
//        System.out.println(levelOrderBottom(root));
//        System.out.println(maxDepth(root));
//        flatten(root);
//        System.out.println(maxDepth(root));
//        ListNode listNode = new ListNode(1,
//                new ListNode(2,
//                        new ListNode(3,
//                                new ListNode(4,
//                                        new ListNode(5,
//                                                new ListNode(6,
//                                                        new ListNode(7)))))));
//
//        TreeNode node = new TreeNode();
//        TreeNode root = node.sortedListToBST(listNode);
//        printTree(root);
//        KthSmallest kthSmallest = new KthSmallest();
//        System.out.println(kthSmallest.kthSmallest(root, 3));
//        System.out.println(inorderTraversal(root));
//        Node root = new Node(1,
//                Arrays.asList(new Node(2,
//                        Arrays.asList(new Node(5),new Node(6))),new Node(3), new Node(4)));
//        System.out.println(root.levelOrder(root));
//        System.out.println(postorderTraversal(root));
//        RecoverTree recoverTree = new RecoverTree();
        TreeNode root = new TreeNode(3,
                new TreeNode(1, new TreeNode(2), new TreeNode(5)),
                new TreeNode(4, new TreeNode(2), null));
        TreeNode treeNode = lowestCommonAncestor(root, root.left.right, root.left);
        System.out.println(treeNode.val);
//        printTree(root);
//        recoverTree.recoverTree(root);
//        System.out.println();
//        printTree(root);
//        SumNumbers sumNumbers = new SumNumbers();
//        System.out.println(sumNumbers.sumNumbers(root));
    }



}

