package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class ProductInfo_toString_9_0_Test {

    @Test
    public void testToString() {
        // Arrange
        ProductInfo productInfo = mock(ProductInfo.class);
        when(productInfo.getListName()).thenReturn("Sample List");
        when(productInfo.getTotalResults()).thenReturn("12345");
        when(productInfo.getTotalPages()).thenReturn("67890");
        when(productInfo.getProductsArrayList()).thenReturn(new ArrayList<>());
        // Act
        String result = productInfo.toString();
        // Assert
        assertEquals("Total results = 12345\nTotal pages = 67890\n# of products = 0", result);
    }
}
