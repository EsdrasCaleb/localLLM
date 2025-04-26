package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class FullProduct_addSimilarItem_7_0_Test {

    @Test
    public void testAddSimilarItem() {
        // Create a mock of MiniProduct
        MiniProduct mockItem = mock(MiniProduct.class);
        // Create an instance of FullProduct
        FullProduct fullProduct = new FullProduct();
        // Call the addSimilarItem method with the mock item
        fullProduct.addSimilarItem(mockItem);
        // Verify that the similarItems list contains the mock item
        assertEquals(1, fullProduct.getSimilarItems().size());
        assertEquals(mockItem, fullProduct.getSimilarItems().get(0));
    }
}
