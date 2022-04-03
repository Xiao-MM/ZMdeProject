package algorithm;

/**
 * 链表结点
 */
public class ListNode {
     int val;
     ListNode next;
     public ListNode() {}
     public ListNode(int val) { this.val = val; }
     public ListNode(int val, ListNode next) { this.val = val; this.next = next; }

     /**
      * 两数相加，不带头结点
      * 1. 两数的长度相同时可以依次从头向下逐个遍历累加
      * 2. 两数长度不同时，依次从低位相加，高位无进位正常写入即可
      * 3. 某一位的值 = p1 + p2 + carry
      * @param l1
      * @param l2
      * @return
      */
     public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
          ListNode head = new ListNode();//作为头结点
          ListNode p = head;// 移动指针
          if (l1 == null) return l2;
          if (l2 == null) return l1;
          int carry = 0;
          int value;
          while (l1 != null || l2 != null){
              if (l1 == null){
                    value = (l2.val + carry)%10;
                    carry = (l2.val + carry)/10;
                    p.next = new ListNode(value);
                    p = p.next;
                    l2 = l2.next;
              }else if (l2 == null){
                    value = (l1.val + carry)%10;
                    carry = (l1.val + carry)/10;
                    p.next = new ListNode(value);
                    p = p.next;
                    l1 = l1.next;
              }else {
                   value = (l1.val + l2.val + carry)%10;
                   carry = (l1.val + l2.val + carry)/10;
                   p.next = new ListNode(value);
                   p = p.next;
                   l1 = l1.next;
                   l2 = l2.next;
              }
              if ((l1 == null && l2 == null) && carry != 0){
                   p.next = new ListNode(carry);
                   p = p.next;
              }

          }
          return head.next;
     }

     public static void printNodes(ListNode l){
          while (l != null){
               System.out.print(l.val + " ");
               l = l.next;
          }
          System.out.println();
     }

    /**
     * 19. 删除链表的倒数第 N 个结点
     * @param head
     * @param n
     * @return
     */
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(-1, head);// 自定义头结点
        if (head == null) return null;
        ListNode pre = dummy, p = dummy;
        int count = 0;
        while (count <= n && p != null){
            p = p.next;
            count ++;
            // 如果倒数n > 链表长度直接返回
            if (p == null && count < n){
                return head;
            }
        }
        while (p != null){
            p = p.next;
            pre = pre.next;
        }
        pre.next = pre.next.next;
        return dummy.next;
    }
//      思路有问题，边界条件处理不了
//    /**
//     * 86. 分隔链表
//     * @param head
//     * @param x
//     * @return
//     */
//    public static ListNode partition(ListNode head, int x) {
//        ListNode dummy = new ListNode(-1, head);
//        if (head == null || head.next == null){
//            return head;
//        }
//        ListNode pre,p,q;
//        pre = q = dummy;
//        p = dummy.next;
//        if (pre.next.val < x){
//            pre = pre.next;
//            p = p.next;
//            q = q.next;
//        }
//        while (p != null){
//            // p的值<x且前面有>x得结点存在需要将其前移
//            if (p.val < x && pre.next.val > x){
//                q.next = p.next;
//                p.next = pre.next;
//                pre.next = p;
//                pre = pre.next;
//                q = q.next;
//                if (q == null){
//                    p = null;
//                }else {
//                    p = q.next;
//                }
//            }else {
//                p = p.next;
//                q = q.next;
//            }
//        }
//        return dummy.next;
//    }

    /**
     //     * 86. 分隔链表
     //     * @param head
     //     * @param x
     //     * @return
     //     */
    public static ListNode partition(ListNode head, int x) {
        ListNode smallHead = new ListNode(-1);
        ListNode small = smallHead;
        ListNode bigHead = new ListNode(-1);
        ListNode big = bigHead;
        while (head != null){
            if (head.val < x){
                small.next = head;
                small = small.next;
            }else {
                big.next = head;
                big = big.next;
            }
            head = head.next;
        }
        big.next = null;
        small.next = bigHead.next;
        return smallHead.next;
    }

    /**
     * 82. 删除排序链表中的重复元素 II
     * @param head
     * @return
     */
    public static ListNode deleteDuplicates(ListNode head) {
        // head为空或者只有一个结点可以直接返回
        if (head == null || head.next == null) {
            return head;
        }
        // 哑结点，避免头被删掉
        ListNode dummy = new ListNode(-1, head);
        ListNode p = dummy;
        int x;
        while (p.next != null && p.next.next != null){
            // 检测到重复元素就记录下当前元素
            if (p.next.val == p.next.next.val){
                x = p.next.val;
                // 一个个删掉重复元素
                while (p.next != null && p.next.val == x){
                    p.next = p.next.next;
                }
            }else {
                p = p.next;
            }
        }
        return dummy.next;
    }

    /**
     * 92. 反转链表 II
     * 头插法
     * @param head
     * @param left
     * @param right
     * @return
     */
    public static ListNode reverseBetween(ListNode head, int left, int right) {
        // 单个元素无须翻转
        if (left == right){
            return head;
        }
        ListNode dummy = new ListNode(-1, head);
        ListNode pre = dummy, p = dummy.next;
        // pre 位于 left 前一个位置，p 始终指向left一开始位于的那个位置
        for (int i = 0; i < left - 1; i++) {
            pre = pre.next;
            p = p.next;
        }

        // 将 (right - left - 1) 个元素一个个以头插法的形式插入pre的后面，p 指向移动元素的下一个元素
        for (int i = 0; i < right - left; i++) {
            ListNode removed = p.next;
            p.next = p.next.next;
            removed.next = pre.next;
            pre.next = removed;
        }

        return dummy.next;
    }

    /**
     * 61. 旋转链表
     * @param head
     * @param k
     * @return
     */
    public static ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        int length = getLength(head);
        if (length == 1) return head;
        // 当 k > length 时等价于对其余数次旋转
        k = k % length;
        if (k == 0) return head;
        int count = 0;
        ListNode p = head, pre = head;
        while (count < k){
            count++;
            p = p.next;
        }
        while (p.next != null){
            p = p.next;
            pre = pre.next;
        }
        ListNode q = pre.next;
        pre.next = null;
        p.next = head;
        return q;
    }

    /**
     * 获取链表长度
     * @param head
     * @return
     */
    private static int getLength(ListNode head){
        int length = 0;
        while (head != null){
            length++;
            head = head.next;
        }
        return length;
    }

    /**
     * 147. 对链表进行插入排序
     * @param head
     * @return
     */
    public static ListNode insertionSortList(ListNode head) {
        if (head == null) return null;
        ListNode dummy = new ListNode();
        dummy.next = head;
        // lastNode 记录排好序的最后一个节点，p 指向待排序的元素
        ListNode lastNode = head, p = head.next;
        while (p != null){
            // p.val 大于最后节点元素值 lastNode 后移
            if (p.val >= lastNode.val){
                lastNode = lastNode.next;
            }else {
                // pre 待插入位置的前驱节点，初始从dummy开始出发
                ListNode pre = dummy;
                // 找到待插入位置的前驱节点
                while (pre.next.val <= p.val) pre = pre.next;
                lastNode.next = p.next;
                p.next = pre.next;
                pre.next = p;
            }
            // p 下移下一个待排元素
            p = lastNode.next;
        }
        return dummy.next;
    }

    /**
     * 141. 环形链表
     * @param head
     * @return
     */
    public static boolean hasCycle(ListNode head) {
        if (head == null) return false;
        ListNode slow = head, fast = head.next;
        while (slow != null && fast != null && fast.next != null){
            if (slow == fast) return true;
            slow = slow.next;
            fast = fast.next.next;
        }
        return false;
    }

    /**
     * 142. 环形链表 II
     * 2*(x+y) = x+y+n(y+z)
     * x = (n-1)*(y+z) + z
     * @param head
     * @return
     */
    public static ListNode detectCycle(ListNode head) {
        if (head == null) return null;
        ListNode slow = head, fast = head, p = null;
        while (slow != null && fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            // 记录下重叠的位置
            if (fast == slow){
                p = fast;
                break;
            }
        }
        // 走完了都没有找到重合位置无环
        if (p == null) return null;
        // head开始起步向下走，p开始从重叠位置起步走，二者重合的节点即为环入口
        while (head != p){
            p = p.next;
            head = head.next;
        }
        return p;
    }

    static class ReorderList{
        /**
         * 143. 重排链表
         * @param head
         */
        public void reorderList(ListNode head) {
            ListNode mid = findMid(head);
            ListNode r = mid.next;
            mid.next = null;
            ListNode reverse = reverse(r);
            merge(head, reverse);
        }

        /**
         * 查找mid
         * @param head
         * @return
         */
        public ListNode findMid(ListNode head){
            if (head == null) return null;
            ListNode slow = head, fast = head;
            while (fast != null && fast.next != null){
                slow = slow.next;
                fast = fast.next.next;
            }
            return slow;
        }

        /**
         * 翻转链表
         * @param head
         * @return
         */
        public ListNode reverse(ListNode head){
            if (head == null) return null;
            ListNode dummy = new ListNode();
            ListNode p = head, q;
            while (p != null){
                q = p.next;
                p.next = dummy.next;
                dummy.next = p;
                p = q;
            }
            return dummy.next;
        }

        /**
         * 合并链表
         * @param l
         * @param r
         * @return
         */
        public void merge(ListNode l, ListNode r){
            ListNode p = l;
            while (p != null && r != null){
                ListNode t1 = p.next;
                ListNode t2 = r.next;
                p.next = r;
                r.next = t1;
                p = t1;
                r = t2;
            }
        }

    }


    /**
     * 24. 两两交换链表中的节点
     * @param head
     * @return
     */
    public static ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode dummy = new ListNode(0, head);
        // p,q 交换 pre记录已做交换列表的最后一个元素
        ListNode p = head, q = head.next, pre = dummy;
        while (p != null && q != null){
            // 交换p,q
            p.next = q.next;
            q.next = p;
            // 交换后链接在已经交换列表的尾部
            pre.next = q;
            // 更新尾部
            pre = p;
            // p,q 继续向后走
            p = p.next;
            if (p != null) q = p.next;
        }
        return dummy.next;
    }

    static class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }

        /**
         * 138. 复制带随机指针的链表
         * @param head
         * @return
         */
        public Node copyRandomList(Node head) {
            if (head == null) return null;
            // 为每一个节点创建副本并且连起来
            Node p = head;
            while (p != null){
                Node q = new Node(p.val);
                q.next = p.next;
                p.next = q;
                p = q.next;
            }
            // 将每个节点随机指针连接起来
            p = head;
            while (p != null && p.next != null){
                if (p.random != null){
                    p.next.random = p.random.next;
                }
                p = p.next.next;
            }
            // 复原源链表
            p = head;
            Node q = head.next;
            Node dummy = new Node(-1);
            dummy.next = q;
            while (q.next != null){
                p.next = q.next;
                p = p.next;
                q.next = p.next;
                q = q.next;
            }
            p.next = null;//这个要断开，不然原链表结构发生变化
            return dummy.next;
        }

        /**
         * 打印链表
         * @param head
         */
        public static void printNodes(Node head){
            Node p = head;
            while (p != null){
                System.out.print(p.val + " ");
                p = p.next;
            }
            System.out.println();
            p = head;
            while (p != null){
                if (p.random != null){
                    System.out.print(p.random.val + " ");
                }else {
                    System.out.print("  ");
                }
                p = p.next;
            }
            System.out.println();
        }
    }

    /**
     * 148. 排序链表
     * @param head
     * @return
     */
    public static ListNode sortList(ListNode head) {
        return sort(head);
    }

    /**
     * 归并排序
     * @param head
     * @return
     */
    private static ListNode sort(ListNode head){
        if (head != null){
            // 单个元素就无须再去找mid，否则会陷入死循环
            if (head.next == null) return head;
            // 查找中间节点
            ListNode mid = findMid(head);
            // 中间节点的下一个节点部分作为右半部分
            ListNode rightPart = mid.next;
            // 从中间断开，中间节点的next置空
            mid.next = null;
            // 递归执行左半部分
            ListNode left = sort(head);
            // 递归执行右半部分
            ListNode right = sort(rightPart);
            // 合并左链表和右链表
            return merge(left, right);
        }
        return null;
    }

    /**
     * 返回中值
     * @param head
     * @return
     */
    private static ListNode findMid(ListNode head){
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    /**
     * 合并链表
     * @param left
     * @param right
     * @return
     */
    private static ListNode merge(ListNode left, ListNode right){
        ListNode dummy = new ListNode();
        ListNode p = dummy;
        while (left != null && right != null){
            if (left.val < right.val){
                p.next = left;
                left = left.next;
            }else {
                p.next = right;
                right = right.next;
            }
            p = p.next;
        }
        // 左链表有剩余直接加入归并链表尾部
        if (left != null) p.next = left;
        // 右链表有剩余直接加入归并链表尾部
        if (right != null) p.next = right;
        // 返回归并后的链表
        return dummy.next;
    }

     public static void main(String[] args) {
//          ListNode l1 = new ListNode(9,new ListNode(0,new ListNode(3)));
//          ListNode l2 = new ListNode(1,new ListNode(5,new ListNode(7)));
//          System.out.println("l1 :");
//          printNodes(l1);
//          System.out.println("l2 :");
//          printNodes(l2);
//          ListNode listNode = addTwoNumbers(l1, l2);
//          System.out.println("sum :");
//          printNodes(listNode);
//         ListNode listNode = new ListNode(1,new ListNode(2, new ListNode(3,new ListNode(4, new ListNode(5)))));
//         ListNode listNode = new ListNode(3,new ListNode(5));
//         printNodes(removeNthFromEnd(listNode, 3));
//         ListNode listNode = new ListNode(2,
//                                 new ListNode(2,
//                                         new ListNode(2,
//                                                 new ListNode(2,
//                                                         new ListNode(2)))));
//         listNode.next.next.next.next.next.next = listNode.next;
//         ListNode listNode = new ListNode(1, new ListNode(2));
//         printNodes(listNode);
//         printNodes(reverseBetween(listNode,2,5));
//         ListNode head = rotateRight(listNode, 2);
//         ListNode listNode1 = insertionSortList(listNode);
//         printNodes(listNode1);
//         System.out.println(hasCycle(listNode));
//         printNodes(swapPairs(listNode));
         ReorderList reorderList = new ReorderList();
//         ListNode reverse = reorderList.reverse(listNode);
         ListNode listNode2 = new ListNode(3,
                                 new ListNode(2,
                                         new ListNode(1,
                                                 new ListNode(5,
                                                         new ListNode(6,
                                                                 new ListNode(4))))));
//         reorderList.reorderList(listNode2);
//         printNodes(listNode2);
//
//         Node node1 = new Node(1);
//         Node node2 = new Node(2);
//         Node node3 = new Node(3);
//         Node node4 = new Node(4);
//         node1.next = node2;
//         node2.next = node3;
//         node3.next = node4;
//         node1.random = node3;
//         node2.random = node2;
//         node4.random = node1;
//         Node.printNodes(node1);
//         Node node = node1.copyRandomList(node1);
//         Node.printNodes(node);
//         Node.printNodes(node1);
//         System.out.println(findMid(listNode2).val);
         ListNode listNode1 = sortList(listNode2);
         printNodes(listNode1);
     }
 }
