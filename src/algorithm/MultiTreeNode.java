package algorithm;

import java.util.*;

/**
 * 多叉树
 */
public class MultiTreeNode {
    public int val;
    public List<MultiTreeNode> children;
    public MultiTreeNode(){}
    public MultiTreeNode(int val, List<MultiTreeNode> children){
        this.val = val;
        this.children = children;
    }

    /**
     * 多叉树的后序遍历算法
     * @param root
     * @return
     */
    public static List<Integer> postOrder(MultiTreeNode root){
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Stack<MultiTreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            MultiTreeNode multiTreeNode = stack.pop();
            result.add(multiTreeNode.val);
            if (multiTreeNode.children != null) {
                for (MultiTreeNode child : multiTreeNode.children) {
                    stack.push(child);
                }
            }
        }
        Collections.reverse(result);
        return result;
    }

    public static void main(String[] args) {
        MultiTreeNode multiTreeNode7 = new MultiTreeNode(7,null);
        MultiTreeNode multiTreeNode6 = new MultiTreeNode(6,new ArrayList<>(Collections.singletonList(multiTreeNode7)));
        MultiTreeNode multiTreeNode5 = new MultiTreeNode(5,null);
        MultiTreeNode multiTreeNode3 = new MultiTreeNode(3,new ArrayList<>(Arrays.asList(multiTreeNode5, multiTreeNode6)));
        MultiTreeNode multiTreeNode2 = new MultiTreeNode(2,null);
        MultiTreeNode multiTreeNode4 = new MultiTreeNode(4,null);
        MultiTreeNode multiTreeNode1 = new MultiTreeNode(1,new ArrayList<>(Arrays.asList(multiTreeNode3, multiTreeNode2, multiTreeNode4)));
        List<Integer> result = postOrder(multiTreeNode1);
        System.out.println(result);
    }

}
