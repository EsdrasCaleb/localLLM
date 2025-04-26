package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_printProductList_5_0_Test {

    private ProductLine productLine;

    private ProductInfo productInfo;

    @BeforeEach
    public void setup() {
        productLine = new ProductLine();
        productInfo = Mockito.mock(ProductInfo.class);
        productLine.setProductInfo(productInfo);
    }

    @Test
    public void testPrintProductList() {
        String mockProductList = "Mock Product List";
        Mockito.when(productInfo.printProductList()).thenReturn(mockProductList);
        String result = productLine.printProductList();
        assertEquals("Mode = null\nMock Product List\n", result);
    }
}
