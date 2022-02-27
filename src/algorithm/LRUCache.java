package algorithm;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 146. LRU 缓存
 */
public class LRUCache extends LinkedHashMap<Integer,Integer> {

    private final int capacity;

    public LRUCache(int capacity) {
        super(capacity, 0.75f, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.get(key) == null ? -1 : super.get(key);
    }

    public void put(int key, int value) {
        super.put(key, value);
    }

    /**
     * 重写该方法使得容量超过缓存容量时移除最近最久未用的节点
     * accessOrder == true 则 void afterNodeInsertion(boolean evict) 会根据该方法的返回值去判断是否移除头结点，
     * 插入的数据是放在链表的尾部的，所以插入的数据永远是最新的
     * accessOrder == true 则 void afterNodeAccess(Node<K,V> p) { } 也会将当前访问的节点放到链表的最后面
     * @param eldest
     * @return
     */
    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity;
    }

    public static void main(String[] args) {
        LRUCache lruCache = new LRUCache(2);
        lruCache.put(1, 1);
        System.out.println(lruCache);
        lruCache.put(2, 2);
        System.out.println(lruCache);
        System.out.println(lruCache.get(1));
        lruCache.put(3, 3);
        System.out.println(lruCache);
        System.out.println(lruCache.get(2));
        lruCache.put(4, 4);
        System.out.println(lruCache);
        System.out.println(lruCache.get(1));
        System.out.println(lruCache.get(3));
        System.out.println(lruCache.get(4));
    }
}
