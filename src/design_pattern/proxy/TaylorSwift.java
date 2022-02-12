package design_pattern.proxy;

/**
 * 歌手泰勒斯威夫特
 */
public class TaylorSwift implements Singer {
    @Override
    public void sing() {
        System.out.println("泰勒正在唱歌...");
    }
}
