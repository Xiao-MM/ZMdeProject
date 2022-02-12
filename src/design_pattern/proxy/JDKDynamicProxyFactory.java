package design_pattern.proxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

/**
 * JDK动态代理，需要被代理有实现接口
 * 下面代码负责产生代理对象
 */
public class JDKDynamicProxyFactory {
    /**
     * 要被代理的对象
     */
    private Object object;

    public JDKDynamicProxyFactory(Object object){
        this.object = object;
    }

    /**
     * 获取代理对象
     * @return
     */
    public Object getProxyInstance(){
        return Proxy.newProxyInstance(
                object.getClass().getClassLoader(),
                object.getClass().getInterfaces(),
                (proxy, method, args) -> {
                    System.out.println("收完钱安排演唱会...");
                    Object o = method.invoke(object, args);
                    System.out.println("安排演唱会事后事宜...");
                    return o;
                });
    }
}
