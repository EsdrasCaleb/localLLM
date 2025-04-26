package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductLine_toString_4_0_Test {

    private ProductLine productLine;

    @BeforeEach
    public void setUp() {
        productLine = new ProductLine();
        productLine.setMode("Production");
        ProductInfo productInfo = mock(ProductInfo.class);
        when(productInfo.toString()).thenReturn("Product Information");
        productLine.setProductInfo(productInfo);
    }

    @Test
    public void testToString() {
        assertEquals("Mode = Production\nProduct Information", productLine.toString());
    }
}
