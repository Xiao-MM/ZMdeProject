package design_pattern;

public class LowMaskFactoryMethod implements FactoryMethod {
    @Override
    public Mask makeMask() {
        return new LowMask();
    }
}
