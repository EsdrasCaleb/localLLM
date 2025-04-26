package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ProductLine_toString_4_2_Test {

    @Mock
    private ProductInfo productInfo;

    @InjectMocks
    private ProductLine productLine;

    @Test
    public void testToString() {
        productLine.setMode("Test Mode");
        productLine.setProductInfo(productInfo);
        String expected = "Mode = Test Mode\n" + productInfo;
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }
}
