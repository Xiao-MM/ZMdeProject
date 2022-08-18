package algorithm;

import java.util.HashMap;
import java.util.Map;

/**
 * 155. 最小栈
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(val);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
public class MinStack {

    /**
     * stack[0]存放当前元素，stack[1]存放栈中的最小元素
     */
    int[][] stack;
    /**
     * 始终指向栈顶元素
     */
    int index = -1;

    public MinStack() {
        stack = new int[16][2];
    }

    public void push(int val) {
        // 需要扩容
        if (index == stack.length - 1){
            // 扩容一倍
            int[][] stackCopy = new int[stack.length << 1][2];
            System.arraycopy(stack, 0, stackCopy, 0, stack.length);
            stack = stackCopy;
        }
        int min = index == -1 ? val : Math.min(stack[index][1], val);
        index++;
        stack[index][0] = val;
        stack[index][1] = min;
    }

    public void pop() {
        if (index == -1) return;
//        stack[index][0] = 0;
//        stack[index][1] = 0;
        index--;
    }

    public int top() {
        return stack[index][0];
    }

    public int getMin() {
        return stack[index][1];
    }
}

