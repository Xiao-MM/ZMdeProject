package design_pattern;

public class HighMaskFactoryMethod implements FactoryMethod {
    @Override
    public Mask makeMask() {
        return new HighMask();
    }
}
