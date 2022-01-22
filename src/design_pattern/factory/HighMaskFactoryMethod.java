package design_pattern.factory;

public class HighMaskFactoryMethod implements FactoryMethod {
    @Override
    public Mask makeMask() {
        return new HighMask();
    }
}
