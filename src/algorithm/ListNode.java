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
         ListNode listNode = new ListNode(1);
         printNodes(removeNthFromEnd(listNode, 3));
     }
 }
