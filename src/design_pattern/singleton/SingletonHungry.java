package design_pattern.singleton;

/**
 * 恶汉式
 */
public class SingletonHungry {

//    private static transient SingletonHungry instance = new SingletonHungry();// 防止反序列化

    private static SingletonHungry instance = new SingletonHungry();

    private SingletonHungry(){}

    public static SingletonHungry getInstance(){
        return instance;
    }
}
