package algorithm;

public class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }

    /**
     * 116. 填充每个节点的下一个右侧节点指针
     * @param root
     * @return
     */
    public static Node connect(Node root) {
        if (root == null) return null;
        // 依次沿着左子树向右串联
        Node pre = root;
        while (pre.left != null){
            Node tmp = pre;
            while (tmp != null){
                // 串联左右子树
                tmp.left.next = tmp.right;
                if (tmp.next != null){
                    // 串联右子树和邻居的左子树
                    tmp.right.next = tmp.next.left;
                }
                // 水平查找下一个要串联的节点
                tmp = tmp.next;
            }
            // 下一行左边起始再开始
            pre = pre.left;
        }
        return root;
    }


    /**
     * 116. 填充每个节点的下一个右侧节点指针
     * @param root
     * @return
     */
    public Node connectDFS(Node root) {
        dfs(root);
        return root;
    }

    private void dfs(Node root){
        if (root == null) return;
        Node left = root.left;
        Node right = root.right;
        // 依次纵向串联中间的相邻节点
        while (left != null){
            left.next = right;
            left = left.right;
            right = right.left;
        }
        dfs(root.left);
        dfs(root.right);
    }

    /**
     * 117. 填充每个节点的下一个右侧节点指针 II
     * @param root
     * @return
     */
    public Node connect2(Node root) {
        if (root == null) return null;
        // 工作指针
        Node p = root;
        // 依次对所有节点进行子节点串联
        while (p != null){
            // 定义哑结点
            Node dummy = new Node();
            // pre 用于记录待被串联节点的前驱结点
            Node pre = dummy;
            // p 在同一层移动，走到底跳出
            while (p != null){
                // p 的左节点不为空，串联左节点
                if (p.left != null){
                    pre.next = p.left;
                    pre = pre.next;
                }
                // // p 的右节点不为空，串联右节点
                if (p.right != null){
                    pre.next = p.right;
                    pre = pre.next;
                }
                // p 移动至同层下一个节点
                p = p.next;
            }
            // 同一层走完，步入下一层开始
            p = dummy.next;
        }
        return root;
    }

}
