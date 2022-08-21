package algorithm;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Schedule {

//    // 采用二维数组来表示邻接表存储边的信息
//    List<List<Integer>> edges;
//    // 标志位默认表示是无环的
//    boolean valid = true;
//    // 访问标志位 0-未访问 1-正在访问 2-访问结束不再进行访问
//    int[] visited;
//
//    /**
//     * 207. 课程表
//     * dfs
//     * @param numCourses
//     * @param prerequisites
//     * @return
//     */
//    public boolean canFinish(int numCourses, int[][] prerequisites) {
//        edges = new ArrayList<>();
//        for (int i = 0; i < numCourses; i++) {
//            edges.add(new ArrayList<>());
//        }
//        visited = new int[numCourses];
//        for (int[] prerequisite : prerequisites) {
//            // 将其邻边加入邻接表中
//            edges.get(prerequisite[1]).add(prerequisite[0]);
//        }
//        for (int i = 0; i < numCourses && valid; i++) {
//            if (visited[i] == 0)
//                dfs(edges, visited, i);
//        }
//        return valid;
//    }
//
//    private void dfs(List<List<Integer>> edges, int[] flag, int i){
//        // 置当前节点正在访问中
//        flag[i] = 1;
//        // 遍历其邻居节点
//        for (Integer j : edges.get(i)) {
//            // 如果邻居未被访问则访问
//            if (visited[j] == 0)
//                dfs(edges, flag, j);
//                // 递归完发现状态变了直接return
//                if (!valid){
//                    return;
//                }
//            // 邻居正在被访问，发生环现象，直接返回，valid 变为false
//            else if (visited[j] == 1){
//                valid = false;
//                return;
//            }
//            // 邻居已经被访问过，直接找下一个邻居
//        }
//        // 回溯时置当前节点已经被访问过
//        flag[i] = 2;
//    }

//    /**
//     * 207. 课程表
//     * bfs
//     * @param numCourses
//     * @param prerequisites
//     * @return
//     */
//    public boolean canFinish(int numCourses, int[][] prerequisites) {
//        // 采用二维数组来表示邻接表存储边的信息
//        List<List<Integer>> edges = new ArrayList<>();
//        // 入度数组
//        int[] inDegree = new int[numCourses];
//        for (int i = 0; i < numCourses; i++) {
//            edges.add(new ArrayList<>());
//        }
//        for (int[] prerequisite : prerequisites) {
//            // 将其邻边加入邻接表中
//            edges.get(prerequisite[1]).add(prerequisite[0]);
//            // prerequisite[0]->prerequisite[1]，每出现一个prerequisite元素意味着某个顶点度数加1
//            inDegree[prerequisite[0]]++;
//        }
//        Queue<Integer> queue = new LinkedList<>();
//        for (int i = 0; i < numCourses; i++) {
//            if (inDegree[i] == 0) queue.offer(i);
//        }
//        // 记录已经被记录的度数为0的节点
//        int count = 0;
//        while (!queue.isEmpty()){
//            Integer p = queue.poll();
//            count++;// 每出队一次意味着度数为0的节点数量+1
//            // 将p的所有邻居入度-1
//            for (Integer i : edges.get(p)) {
//                if (--inDegree[i] == 0){
//                    queue.offer(i);
//                }
//            }
//        }
//        // 若度为0的节点的个数等于numCourses则表示无环
//        return count == numCourses;
//    }



    /**
     * 207. 课程表
     * 原数组下手，统计出度
     * [[0,1],[0,3],[1,2],[1,3],[2,4],[3,4]]
     * [[1,0],[3,0],[2,1],[3,1],[4,2],[4,3]]
     * [2,2,1,1,0]
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] degree = new int[numCourses];
        int length = prerequisites.length;
        for (int[] prerequisite : prerequisites) {
            // 统计出度
            degree[prerequisite[1]]++;
        }
        boolean[] isRemoved = new boolean[length];
         int whileCount = 0;
        int forCount = 0;
        int removed = 0;
        while (removed < length){
            // 当前移除数量
            int curRemoved = 0;
            for (int i = 0; i < length; i++) {
                // 已经被移除了直接进入下一个循环
                if (isRemoved[i]) continue;
                int[] p = prerequisites[i];
                // 出度为0的点被移除
                if (degree[p[0]] == 0){
                    // 指向其的邻居节点出度-1
                    degree[p[1]]--;
                    isRemoved[i] = true;
                    curRemoved++;
                    forCount++;
                }
            }
            // 一轮循环走完没有元素被移除，则有环，直接return false
            if (curRemoved == 0) return false;
            removed += curRemoved;
            whileCount++;
        }
        System.out.println("whileCount = "+whileCount);
        System.out.println("forCount = "+forCount);
        return removed == length;
    }

//    /**
//     * 207. 课程表
//     * 原数组下手，统计入度
//     * [[0,1],[0,3],[1,2],[1,3],[2,4],[3,4]]
//     * [[1,0],[3,0],[2,1],[3,1],[4,2],[4,3]]
//     * [0,1,1,2,2]
//     * @param numCourses
//     * @param prerequisites
//     * @return
//     */
//    public boolean canFinish(int numCourses, int[][] prerequisites) {
//        int[] degree = new int[numCourses];
//        int length = prerequisites.length;
//        for (int[] prerequisite : prerequisites) {
//            // 统计入度
//            degree[prerequisite[0]]++;
//        }
//        boolean[] isRemoved = new boolean[length];
//        int whileCount = 0;
//        int forCount = 0;
//        int removed = 0;
//        while (removed < length){
//            // 当前移除数量
//            int curRemoved = 0;
//            for (int i = 0; i < length; i++) {
//                // 已经被移除了直接进入下一个循环
//                if (isRemoved[i]) continue;
//                int[] p = prerequisites[i];
//                // 入度为0的点被移除
//                if (degree[p[1]] == 0){
//                    // 指向其的邻居节点入度-1
//                    degree[p[0]]--;
//                    isRemoved[i] = true;
//                    curRemoved++;
//                    forCount++;
//                }
//            }
//            // 一轮循环走完没有元素被移除，则有环，直接return false
//            if (curRemoved == 0) return false;
//            removed += curRemoved;
//            whileCount++;
//        }
//        System.out.println("whileCount = "+whileCount);
//        System.out.println("forCount = "+forCount);
//        return removed == length;
//    }

    public static void main(String[] args) {
        Schedule schedule = new Schedule();
        System.out.println(schedule.canFinish(5, new int[][]{{0,1},{0,3},{1,2},{1,3},{2,4},{3,4}}));
    }

//    /**
//     * 207. 课程表
//     * dfs
//     * @param numCourses
//     * @param prerequisites
//     * @return
//     */
//    public boolean canFinish(int numCourses, int[][] prerequisites) {
//        Node[] nodes = new Node[numCourses];
//        int[] flag = new int[numCourses];
//        for (int i = 0; i < numCourses; i++) {
//            nodes[i] = new Node(i);
//        }
//        // 初始化邻接表
//        for (int[] prerequisite : prerequisites) {
//            Node p = nodes[prerequisite[1]];
//            while (p.next != null) p = p.next;
//            p.next = new Node(prerequisite[0]);
//        }
//
//
//    }


//    public boolean dfs(Node node, int[] flag){
//        if (node == null) return true;
//        if (flag[node.val])
//    }

//    static class Node{
//        int val;
//        Node next;
//        public Node() {};
//        public Node(int val) {
//            this.val = val;
//        }
//        public Node(int val, Node next) {
//            this.val = val;
//            this.next = next;
//        }
//    }


}

