package test;

import entity.User;

public class Test {
    public static void main(String[] args) {
        User user = new User("zs",18);
        System.out.println("引用前："+ user);
        resetName(user);
        System.out.println("引用后："+ user);
        Integer a = 10;
        System.out.println("改变前的值" + a);
        addNum(a);
        System.out.println("改变后的值" + a);
    }

    static void resetName(User user){
        user.setName("--");
    }

    static void addNum(Integer a){
        a += 1;
        System.out.println("方法中：" + a);
    }
}
