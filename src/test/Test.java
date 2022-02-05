package test;

import entity.User;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

public class Test {
    public static void main(String[] args) {
//        User user = new User("zs",18);
//        System.out.println("引用前："+ user);
//        resetName(user);
//        System.out.println("引用后："+ user);
//        Integer a = 10;
//        System.out.println("改变前的值" + a);
//        addNum(a);
//        System.out.println("改变后的值" + a);

//        Map<String,String> map = new HashMap<>();
//        map.put("1","zs");
//        Map<String, String> unmodifiableMap = Collections.unmodifiableMap(map);
//        unmodifiableMap.put("1","ls");
//        Integer[] arr = {1, 2, 3};
//        List list = Arrays.asList(arr);
//        System.out.println(list);
//        int a = 1,b = 3;
//        System.out.println(a + (b-a) >> 1);
//        System.out.println(a + (b-a) / 2);
//        System.out.println(a + ((b-a) >> 1));
//        List<String> list = new ArrayList<>();
//        List<String> synList = Collections.synchronizedList(list);
//        List<Object> copyOnWriteArrayList = new CopyOnWriteArrayList<>();
        HashMap map = new HashMap();
//        System.out.println("K1".hashCode());
        map.put(null,"zs");
        System.out.println(map.get(null));


    }

    static void resetName(User user){
        user.setName("--");
    }

    static void addNum(Integer a){
        a += 1;
        System.out.println("方法中：" + a);
    }


}
