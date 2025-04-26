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
    public void testToString() {
        BlendedSearch blendedSearch = new BlendedSearch();
        assertNull(blendedSearch.toString());
        ProductLine productLine = new ProductLine();
        blendedSearch.getProductLinesArrayList().add(productLine);
        assertNotNull(blendedSearch.toString());
        ProductLine productLine2 = new ProductLine();
        blendedSearch.getProductLinesArrayList().add(productLine2);
        assertNotNull(blendedSearch.toString());
        ProductLine productLine3 = new ProductLine();
        blendedSearch.getProductLinesArrayList().add(productLine3);
        assertNotNull(blendedSearch.toString());
    }
}
