package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FullProduct_printFullProduct_8_0_Test {

    @Test
    void printFullProduct() {
        // Given
        FullProduct fullProduct = new FullProduct();
        ProductDetails details = new ProductDetails();
        ArrayList accessories = new ArrayList();
        ArrayList similarItems = new ArrayList();
        // When
        fullProduct.setDetails(details);
        fullProduct.setAccessories(accessories);
        fullProduct.setSimilarItems(similarItems);
        // Then
        assertDoesNotThrow(() -> fullProduct.printFullProduct());
        // Verify details
        assertNotNull(fullProduct.getDetails());
        // Verify accessories
        assertNotNull(fullProduct.getAccessories());
        assertEquals(0, fullProduct.getAccessories().size());
        // Verify similar items
        assertNotNull(fullProduct.getSimilarItems());
        assertEquals(0, fullProduct.getSimilarItems().size());
    }
}
