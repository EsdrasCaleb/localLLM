package net.kencochrane.a4j.beans;

import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ProductInfo_toString_9_0_Test {

    @Test
    public void testToString() {
        // Arrange
        ProductInfo productInfo = new ProductInfo();
        // Act
        String output = productInfo.toString();
        // Assert
        assertEquals("Total results = 1", productInfo.getTotalResults(), 0);
        assertEquals("Total pages = 2", productInfo.getTotalPages(), 0);
        assertEquals("listName = Product Details", productInfo.getListName(), 0);
        // If total products are null, expected behavior
        if (productInfo.getProductsArrayList() != null) {
            assertEquals("products is null", productInfo.getProductsArrayList().isEmpty(), 0);
        }
    }
}
