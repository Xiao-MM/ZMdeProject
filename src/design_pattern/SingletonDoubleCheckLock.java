package design_pattern;

/**
 * 双重检测锁
 */
public class SingletonDoubleCheckLock {

//    private static volatile transient SingletonDoubleCheckLock singletonDoubleCheckLock;
    /**
     * volatile 避免指令重排
     */
    private static volatile SingletonDoubleCheckLock singletonDoubleCheckLock;

    private SingletonDoubleCheckLock(){}

    public static SingletonDoubleCheckLock getInstance(){
        if (singletonDoubleCheckLock == null){// 第一层检测避免了synchronized影响每次访问的效率
            synchronized (SingletonDoubleCheckLock.class){
                if (singletonDoubleCheckLock == null){// 第二次检测避免了两个线程进来创建两个对象
                    singletonDoubleCheckLock = new SingletonDoubleCheckLock();
                }
            }
        }
        return singletonDoubleCheckLock;
    }

}
