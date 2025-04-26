package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BlendedSearch_toString_3_0_Test {

    @Test
    void testToString() {
        BlendedSearch blendedSearch = new BlendedSearch();
        ProductLine[] productLines = new ProductLine[1];
        productLines[0] = new ProductLine();
        blendedSearch.setProductLine(productLines);
        String result = blendedSearch.toString();
        assertNotNull(result);
        assertEquals("productLines is null", result);
    }

    @Test
    void testToStringWithNullArrayList() {
        BlendedSearch blendedSearch = new BlendedSearch();
        ProductLine[] productLines = new ProductLine[0];
        blendedSearch.setProductLine(productLines);
        String result = blendedSearch.toString();
        assertNull(result);
    }

    @Test
    void testToStringWithNonObject() {
        BlendedSearch blendedSearch = new BlendedSearch();
        ProductLine[] productLines = new ProductLine[0];
        blendedSearch.setProductLine(productLines);
        String result = blendedSearch.toString();
        assertNull(result);
    }
}
