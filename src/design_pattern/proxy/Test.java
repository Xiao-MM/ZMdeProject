package design_pattern.proxy;

public class Test {
    public static void main(String[] args) {
        Singer taylor = new TaylorSwift();
//        StaticSingerProxy proxy = new StaticSingerProxy(taylor);
//        proxy.sing();
        JDKDynamicProxyFactory factory = new JDKDynamicProxyFactory(taylor);
        Singer proxy = (Singer)factory.getProxyInstance();
        proxy.sing();
    }
}
