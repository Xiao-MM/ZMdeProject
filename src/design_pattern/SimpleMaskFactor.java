package design_pattern;

public class SimpleMaskFactor {
    public Mask makeMask(String type){
        Mask mask = null;
        if ("高端口罩".equals(type)){
            mask = new HighMask();
        }
        if ("低端口罩".equals(type)){
            mask = new LowMask();
        }
        return mask;
    }
}
