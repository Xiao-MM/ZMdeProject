package entity;

/**
 * clone() 是 Object 的 protected 方法，它不是 public，一个类不显式去重写 clone()，其它类就不能直接去调用该类实例的 clone() 方法。
 */
public class CloneExample implements Cloneable{
    private int a;
    private int b;
    @Override
    public CloneExample clone() throws CloneNotSupportedException {
        return (CloneExample)super.clone();
    }

    /**
     * 以上抛出了 CloneNotSupportedException，这是因为 CloneExample 没有实现 Cloneable 接口。
     *
     * 应该注意的是，clone() 方法并不是 Cloneable 接口的方法，而是 Object 的一个 protected 方法。Cloneable 接口只是规定，如果一个类没有实现 Cloneable 接口又调用了 clone() 方法，就会抛出 CloneNotSupportedException。
     * @param args
     * @throws CloneNotSupportedException
     */
    public static void main(String[] args) throws CloneNotSupportedException {
        CloneExample e1 = new CloneExample();
        CloneExample e2 = e1.clone();
        System.out.println(e1);
        System.out.println(e2);
    }
}
