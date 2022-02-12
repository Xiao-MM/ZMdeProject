package design_pattern.proxy;

/**
 * 静态歌手代理人
 * 同样要实现歌手接口，自己虽然不唱但是找歌手唱
 * 控制对目标对象的访问，并增加一些功能，可以不用改变源对象的代码
 */
public class StaticSingerProxy implements Singer{

    /**
     * 被代理的一名歌手
     */
    private Singer singer;

    public StaticSingerProxy(Singer singer){
        this.singer = singer;
    }

    @Override
    public void sing() {
        System.out.println("收完钱安排演唱会...");
        singer.sing();
        System.out.println("安排演唱会事后事宜...");
    }
}
