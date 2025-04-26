package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

class BlendedSearch_toString_3_2_Test {

    @Test
    void testToString_withEmptyProductLines() throws Exception {
        BlendedSearch bs = new BlendedSearch();
        ProductLine[] productLines = {};
        bs.setProductLine(productLines);
        String expectedOutput = "# of productLines = 0\n";
        assertEquals(expectedOutput, bs.toString());
    }

    @Test
    void testToString_withNullProductLines() throws Exception {
        BlendedSearch bs = new BlendedSearch();
        Field field = BlendedSearch.class.getDeclaredField("productLines");
        field.setAccessible(true);
        field.set(bs, null);
        String expectedOutput = "productLines is null \n";
        assertEquals(expectedOutput, bs.toString());
    }
}

class ProductLine implements Serializable {

    private String name;

    public ProductLine() {
    }

    public ProductLine(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return "ProductLine{" + "name='" + name + '\'' + '}';
    }
}

class BlendedSearch {

    private ProductLine[] productLines;

    public void setProductLine(ProductLine[] productLines) {
        this.productLines = productLines;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (productLines == null) {
            sb.append("productLines is null \n");
        } else {
            for (ProductLine pl : productLines) {
                sb.append(pl.toString()).append("\n");
            }
            sb.append("# of productLines = ").append(productLines.length).append("\n");
        }
        return sb.toString();
    }
}
