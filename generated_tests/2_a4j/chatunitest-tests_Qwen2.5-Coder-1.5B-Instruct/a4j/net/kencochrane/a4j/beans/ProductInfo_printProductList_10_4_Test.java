package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class ProductInfo_printProductList_10_4_Test {

    @Test
    public void testPrintProductList() throws Exception {
        // Arrange
        ProductInfo productInfo = mock(ProductInfo.class);
        when(productInfo.getListName()).thenReturn("testList");
        when(productInfo.getTotalResults()).thenReturn("100");
        when(productInfo.getTotalPages()).thenReturn("5");
        when(productInfo.getProductsArrayList()).thenReturn(new ArrayList<>());
        // Act
        String result = productInfo.printProductList();
        // Assert
        assertEquals("Total results = 100\n" + "Total pages = 5\n" + "products is null \n" + "# of products = 0\n", result);
    }
}
