package design_pattern.factory;

public class LowMaskFactoryMethod implements FactoryMethod {
    @Override
    public Mask makeMask() {
        return new LowMask();
    }
}
