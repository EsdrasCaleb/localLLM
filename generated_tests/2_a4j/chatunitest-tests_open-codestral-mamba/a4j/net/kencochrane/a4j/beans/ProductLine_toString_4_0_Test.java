package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_0_Test {

    @Mock
    private ProductInfo productInfo;

    private ProductLine productLine;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        productLine = new ProductLine();
        productLine.setProductInfo(productInfo);
    }

    @Test
    public void testToString() {
        String expected = "Mode = null\n" + productInfo + "\n";
        when(productInfo.toString()).thenReturn("ProductInfo{...}");
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }
}
