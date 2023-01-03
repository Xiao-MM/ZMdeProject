package algorithm;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * 855. 考场就座
 * Your ExamRoom object will be instantiated and called as such:
 * ExamRoom obj = new ExamRoom(n);
 * int param_1 = obj.seat();
 * obj.leave(p);
 */
public class ExamRoom {

    int n;

    // 大顶堆，维持一个间距最大的区间
    PriorityQueue<int[]> priorityQueue = new PriorityQueue<>((o1, o2) -> (o2[1] - o2[0]) - (o1[1] - o1[0]));

    public ExamRoom(int n) {
        this.n = n;
    }

    public int seat() {
        // 取最大间距
        int[] e = priorityQueue.poll();
        // 初次入座
        if (e == null){
            priorityQueue.offer(new int[]{0, n - 1});
            return 0;
        }
        int pos = (e[1] - e[0]) >> 1;
        priorityQueue.offer(new int[]{e[0], pos});
        priorityQueue.offer(new int[]{pos, e[1]});
        return pos;
    }

    public void leave(int p) {

    }

}
