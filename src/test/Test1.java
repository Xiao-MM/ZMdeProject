package test;

import java.util.HashMap;

public class Test1 {
    public static void main(String[] args) {
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

        String s1=new String("kvill");

        String s2=s1.intern();

        System.out.println( s1==s1.intern() );

        System.out.println( s1+" "+s2 );

        System.out.println( s2==s1.intern() );

    }
}
