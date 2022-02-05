package entity;

import java.util.HashSet;
import java.util.Objects;

public class EqualExample {

    private int x;
    private int y;
    private int z;

    public EqualExample(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        EqualExample that = (EqualExample) o;

        if (x != that.x) return false;
        if (y != that.y) return false;
        return z == that.z;
    }

//    @Override
//    public int hashCode() {
//        return Objects.hash(x, y, z);
//    }

    /**
     * 下面的代码中，新建了两个等价的对象，并将它们添加到 HashSet 中。
     * 我们希望将这两个对象当成一样的，只在集合中添加一个对象。
     * 但是 EqualExample 没有实现 hashCode() 方法，
     * 因此这两个对象的哈希值是不同的，最终导致集合添加了两个等价的对象。
     * @param args
     */
    public static void main(String[] args) {
        EqualExample e1 = new EqualExample(1, 1, 1);
        EqualExample e2 = new EqualExample(1, 1, 1);
        System.out.println(e1.hashCode());
        System.out.println(e2.hashCode());
        System.out.println(e1.equals(e2)); // true
        HashSet<EqualExample> set = new HashSet<>();
        set.add(e1);
        set.add(e2);
        System.out.println(set.size());   // 2
    }
}