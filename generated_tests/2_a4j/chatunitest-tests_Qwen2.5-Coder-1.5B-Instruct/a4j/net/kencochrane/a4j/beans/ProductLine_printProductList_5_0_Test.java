package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ProductLine_printProductList_5_0_Test {

    @Mock
    private ProductInfo productInfo;

    @InjectMocks
    private ProductLine productLine;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        when(productInfo.printProductList()).thenReturn("Product List Details");
    }

    @Test
    public void printProductList_ShouldReturnCorrectOutput() {
        String result = productLine.printProductList();
        assertEquals("Product List Details", result);
    }

    @Test
    public void testPrintProductList() {
        String expectedOutput = "Mode = default\nProduct List Details\n";
        assertEquals(expectedOutput, productLine.printProductList());
    }

    @Test
    public void testPrintProductListWithCustomModeAndProductInfo() {
        productLine.setMode("custom");
        when(productInfo.printProductList()).thenReturn("Custom Product List Details");
        String expectedOutput = "Mode = custom\nCustom Product List Details\n";
        assertEquals(expectedOutput, productLine.printProductList());
    }
}
