package test;

import entity.User;

import java.io.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Test1 {
    public static void main(String[] args) throws IOException, ClassNotFoundException, IllegalAccessException, InvocationTargetException, InstantiationException {
//        ClassLoader systemClassLoader = ClassLoader.getSystemClassLoader();
//        System.out.println(systemClassLoader.getClass().getName());
//        Integer x = new Integer(123);
//        Integer y = new Integer(123);
//        System.out.println(x == y);    // false
//        Integer z = Integer.valueOf(123);
//        Integer k = Integer.valueOf(123);
//        System.out.println(z == k);   // true
//        StringBuilder stringBuilder = new StringBuilder();
//        String s1 = new String("aaa");
//        String s2 = new String("aaa");
//        System.out.println(s1 == s2);           // false
//        String s3 = s1.intern();
//        String s4 = s2.intern();
//        System.out.println(s3 == s4);           // true
//        String s5 = "bbb";
//        String s6 = "bbb";
//        System.out.println(s5 == s6);  // true
//        char a = 'a';
//        String s = "aaacaaa";
//        System.out.println(s.charAt(s.length()-1) - 'a');
//        System.out.println((char) (0 + 'a'));
//        System.out.println((char) (1 + 'a'));
//        System.out.println((char) (2 + 'a'));

//        String s1=new String("kvill");
//
//        String s2=s1.intern();
//
//        System.out.println( s1==s1.intern() );
//
//        System.out.println( s1+" "+s2 );
//
//        System.out.println( s2==s1.intern() );
//        backtrack();
//        System.out.println(Runtime.getRuntime().freeMemory());
//        System.out.println(Runtime.getRuntime().maxMemory());
//        System.out.println(Runtime.getRuntime().totalMemory());
//        System.out.println(B.a);
//        List<Integer> list = new ArrayList<>();
//        list.add(1);
//        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(""));
//        oos.defaultWriteObject();
//        oos.writeObject(list);
//        oos.close();
//        ObjectInputStream ois = new ObjectInputStream(new FileInputStream())
//        for (Constructor<?> constructor : User.class.getConstructors()) {
//            User user = (User)constructor.newInstance();
//            user.setName("zs");
//            System.out.println(user);
//        }
//        byte a = 127;
//
//        byte b = 127;
//
//        b = a + b; // error : cannot convert from int to byte
//
//        b += a; // ok
//        System.out.println(b);
        ok:
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                System.out.println("i=" + i + ",j=" + j);
                if (j == 5) {
                    break ok;
                }

            }
        }
        A a = new A();
        A.a();

    }

    static void backtrack(){
        System.out.println("调自己..." );
        backtrack();
        System.gc();
    }




}
class A{
    static int a = 1;
    public static void a(){
        System.out.println(a);
    }
}
class B extends A{

}

interface IA{
    void a();
}
interface IB{
    void b();
}
interface IC extends IA,IB{
    void c();
}

class D implements IA,IB{

    @Override
    public void a() {

    }

    @Override
    public void b() {

    }
}