package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BlendedSearch_toString_3_3_Test {

    @Test
    public void testToString() {
        BlendedSearch blendedSearch = new BlendedSearch();
        blendedSearch.setProductLine(new ProductLine[] { new ProductLine() });
        assertEquals("ProductLine\n" + "# of productLines = 1\n", blendedSearch.toString());
    }
}

class ProductLine {

    private List<ProductLine> productLines;

    public ProductLine() {
        productLines = new ArrayList<>();
    }

    public void addProductLine(ProductLine productLine) {
        productLines.add(productLine);
    }

    @Override
    public String toString() {
        return "ProductLine";
    }
}
