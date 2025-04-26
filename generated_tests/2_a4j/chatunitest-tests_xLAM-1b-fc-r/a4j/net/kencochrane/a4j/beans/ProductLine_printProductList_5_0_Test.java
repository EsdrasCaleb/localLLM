package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ProductLine_printProductList_5_0_Test {

    @Mock
    ProductInfo productInfo;

    @InjectMocks
    ProductLine productLine;

    @Test
    public void testPrintProductList() {
        // Given
        String mode = "Test Mode";
        String productList = "Product 1\nProduct 2\nProduct 3";
        String expectedOutput = "Mode = Test Mode\n" + productList;
        when(productInfo.printProductList()).thenReturn(productList);
        productLine.setMode(mode);
        productLine.setProductInfo(productInfo);
        // When
        String result = productLine.printProductList();
        // Then
        assertEquals(expectedOutput, result);
    }
}
