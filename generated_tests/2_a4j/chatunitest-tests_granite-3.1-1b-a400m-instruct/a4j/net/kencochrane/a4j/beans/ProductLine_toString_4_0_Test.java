package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_0_Test {

    private ProductLine productLine;

    @BeforeEach
    public void setUp() {
        productLine = Mockito.mock(ProductLine.class);
        Mockito.when(productLine.getMode()).thenReturn("Mode 1");
        Mockito.when(productLine.getProductInfo()).thenReturn(Mockito.mock(ProductInfo.class));
    }

    @Test
    public void testToString() {
        ProductInfo productInfo = Mockito.mock(ProductInfo.class);
        Mockito.when(productInfo.toString()).thenReturn("Product Info: Some details");
        String expectedOutput = "Mode = Mode 1\nProduct Info: Some details\n";
        String actualOutput = productLine.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
