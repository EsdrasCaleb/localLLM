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

    @InjectMocks
    private ProductLine productLine;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString() {
        // Arrange
        String mode = "TestMode";
        String productInfoString = "ProductInfoString";
        when(productInfo.toString()).thenReturn(productInfoString);
        productLine.setMode(mode);
        productLine.setProductInfo(productInfo);
        // Act
        String result = productLine.toString();
        // Assert
        String expected = "Mode = " + mode + "\n" + productInfoString + "\n";
        assertEquals(expected, result);
    }
}
