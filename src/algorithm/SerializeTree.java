package algorithm;

public class SerializeTree {

    int index;

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "x";
        return root.val + "(" + serialize(root.left) + ")(" + serialize(root.right) + ")";
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        index = 0;
        return parse(data);
    }

    private TreeNode parse(String data){
        // 如果扫描到x则判定为空节点，直接返回空
        if (data.charAt(index) == 'x') {
            // 指针下移
            index++;
            return null;
        }
        // 创建新节点
        TreeNode node = new TreeNode(parseInt(data));
        node.left = parseSubTree(data);
        node.right = parseSubTree(data);
        return node;
    }

    // 跳过括号返回括号体的子树串
    private TreeNode parseSubTree(String data){
        // 跳过左括号
        index++;
        TreeNode subTree = parse(data);
        // 跳过右括号
        index++;
        return subTree;
    }

    // 将字符转int
    private int parseInt(String data){
        int val = 0, sign = 1;
        // 负数
        if (!Character.isDigit(data.charAt(index))){
            sign = -sign;
            index++;
        }
        // 一直向后扫描获取完整数字
        while (Character.isDigit(data.charAt(index)))
            val = val * 10 + data.charAt(index++) - '0';
        return sign * val;
    }

}
