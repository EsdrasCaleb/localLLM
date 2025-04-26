package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_0_Test {

    @Test
    public void testToString() {
        ProductLine productLine = new ProductLine();
        productLine.setMode("testMode");
        ProductInfo productInfo = mock(ProductInfo.class);
        when(productInfo.toString()).thenReturn("testProductInfo");
        productLine.setProductInfo(productInfo);
        String expected = "Mode = testMode\n" + "testProductInfo\n";
        assertEquals(expected, productLine.toString());
    }
}
