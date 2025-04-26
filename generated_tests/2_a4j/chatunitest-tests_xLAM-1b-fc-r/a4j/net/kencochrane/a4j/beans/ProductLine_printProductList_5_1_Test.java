package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ProductLine_printProductList_5_1_Test {

    @Mock
    private ProductInfo productInfo;

    @InjectMocks
    private ProductLine productLine;

    @Test
    public void testPrintProductList() {
        // Given
        when(productInfo.printProductList()).thenReturn("Product List");
        productLine.setProductInfo(productInfo);
        // When
        String expected = "Mode = test\nProduct List\n";
        // Then
        assertEquals(expected, productLine.printProductList());
    }
}
