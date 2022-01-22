package design_pattern.singleton;

/**
 * 懒汉
 * 线程不安全
 */
public class SingletonLazy {

//    private static transient SingletonLazy instance;

    private static SingletonLazy instance;

    private SingletonLazy(){}

    public static SingletonLazy getInstance(){
        if (instance == null){
            instance = new SingletonLazy();
        }
        return instance;
    }
//    //线程安全但效率低下，不推荐，不管instance是否存在多线程每次进来都要线程阻塞
//    public static synchronized SingletonLazy getInstance(){
//        if (instance == null){
//            instance = new SingletonLazy();
//        }
//        return instance;
//    }

}
