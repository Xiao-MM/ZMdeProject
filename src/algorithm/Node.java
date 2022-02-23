package algorithm;

import java.util.*;

/**
 * 多叉树
 */
public class Node {
    public int val;
    public List<Node> children;
    public Node(){}
    public Node(int val,List<Node> children){
        this.val = val;
        this.children = children;
    }

    /**
     * 多叉树的后序遍历算法
     * @param root
     * @return
     */
    public static List<Integer> postOrder(Node root){
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Stack<Node> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            Node node = stack.pop();
            result.add(node.val);
            if (node.children != null) {
                for (Node child : node.children) {
                    stack.push(child);
                }
            }
        }
        Collections.reverse(result);
        return result;
    }

    public static void main(String[] args) {
        Node node7 = new Node(7,null);
        Node node6 = new Node(6,new ArrayList<>(Collections.singletonList(node7)));
        Node node5 = new Node(5,null);
        Node node3 = new Node(3,new ArrayList<>(Arrays.asList(node5, node6)));
        Node node2 = new Node(2,null);
        Node node4 = new Node(4,null);
        Node node1 = new Node(1,new ArrayList<>(Arrays.asList(node3, node2,node4)));
        List<Integer> result = postOrder(node1);
        System.out.println(result);
    }

}
