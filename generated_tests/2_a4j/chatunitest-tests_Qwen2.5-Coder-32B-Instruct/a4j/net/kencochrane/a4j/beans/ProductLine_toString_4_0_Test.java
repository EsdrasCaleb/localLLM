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

    @Mock
    private ProductInfo productInfo;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        productLine = new ProductLine();
        productLine.setMode("TestMode");
        productLine.setProductInfo(productInfo);
    }

    @Test
    void testToString() {
        // Arrange
        String expectedMode = "TestMode";
        String expectedProductInfoString = "Mocked ProductInfo String";
        when(productInfo.toString()).thenReturn(expectedProductInfoString);
        // Act
        String result = productLine.toString();
        // Assert
        String expectedOutput = "Mode = " + expectedMode + "\n" + expectedProductInfoString + "\n";
        assertEquals(expectedOutput, result);
    }
}
