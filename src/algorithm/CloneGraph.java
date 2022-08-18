package algorithm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CloneGraph {

    static class Node {
        public int val;
        public List<Node> neighbors;
        public Node() {
            val = 0;
            neighbors = new ArrayList<>();
        }
        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<>();
        }
        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }


        /**
         * 133. 克隆图
         * @param node
         * @return
         */
        public static Node cloneGraph(Node node) {
            Map<Node, Node> cloneMap = new HashMap<>();
            return dfs(node, cloneMap);
        }

        /**
         * 封装新复制节点的邻居，返回
         * @param node 当前节点
         * @param cloneMap 节点和邻居的一一映射
         * @return
         */
        public static Node dfs(Node node, Map<Node, Node> cloneMap){
            // 当前节点为空，直接返回空
            if (node == null) return null;
            // 如果当前节点的克隆节点存在直接取出来返回即可
            if (cloneMap.containsKey(node)) return cloneMap.get(node);
            // 当前节点对应的克隆节点不存在，重新创建并更新其邻居
            Node clone = new Node(node.val, new ArrayList<>());
            // 创建节点与克隆节点的一一映射
            cloneMap.put(node, clone);
            // 依次添加克隆邻居
            for (Node neighbor : node.neighbors)
                clone.neighbors.add(dfs(neighbor, cloneMap));
            return clone;
        }
    }


}

