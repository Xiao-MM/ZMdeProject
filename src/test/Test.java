package test;

import entity.User;

import java.lang.ref.PhantomReference;
import java.lang.ref.SoftReference;
import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

public class Test {
    private static Test instance;
//    public static int x = 6;

    static {
        System.out.println("static开始");
        // 下面这句编译器报错，非法向前引用
//         System.out.println("x=" + x);
        instance = new Test();
        System.out.println("static结束");
    }

    public Test() {
        System.out.println("构造器开始");
        System.out.println("x=" + x + ";y=" + y);
        // 构造器可以访问声明于他们后面的静态变量
        // 因为静态变量在类加载的准备阶段就已经分配内存并初始化0值了
        // 此时 x=0，y=0
        x++;
        y++;
        System.out.println("x=" + x + ";y=" + y);
        System.out.println("构造器结束");
    }

    public static int x = 6;
    public static int y;

    public static Test getInstance() {
        return instance;
    }

    public static void main(String[] args) {
        Test obj = Test.getInstance();
        System.out.println("x=" + obj.x);
        System.out.println("y=" + obj.y);
    }
//    static {
//        i = 0;                // 给变量赋值可以正常编译通过
//        System.out.print(i);  // 这句编译器会提示“非法向前引用”
//    }
//    static int i = 1;

//    public static void main(String[] args) {
////        User user = new User("zs",18);
////        System.out.println("引用前："+ user);
////        resetName(user);
////        System.out.println("引用后："+ user);
////        Integer a = 10;
////        System.out.println("改变前的值" + a);
////        addNum(a);
////        System.out.println("改变后的值" + a);
//
////        Map<String,String> map = new HashMap<>();
////        map.put("1","zs");
////        Map<String, String> unmodifiableMap = Collections.unmodifiableMap(map);
////        unmodifiableMap.put("1","ls");
////        Integer[] arr = {1, 2, 3};
////        List list = Arrays.asList(arr);
////        System.out.println(list);
////        int a = 1,b = 3;
////        System.out.println(a + (b-a) >> 1);
////        System.out.println(a + (b-a) / 2);
////        System.out.println(a + ((b-a) >> 1));
        List<String> list = new ArrayList<>();
        List<String> synList = Collections.synchronizedList(list);
        List<Object> copyOnWriteArrayList = new CopyOnWriteArrayList<>();
////        HashMap map = new HashMap();
////        System.out.println("K1".hashCode());
////        map.put(null,"zs");
////        System.out.println(map.get(null));
//
//    }
//
//    static void resetName(User user){
//        user.setName("--");
//    }
//
//    static void addNum(Integer a){
//        a += 1;
//        System.out.println("方法中：" + a);
//    }


}
