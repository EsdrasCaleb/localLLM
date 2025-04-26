package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class FullProduct_addAccessory_6_1_Test {

    @Mock
    private MiniProduct miniProduct;

    @Test
    public void testAddAccessory() {
        // Arrange
        FullProduct fullProduct = Mockito.mock(FullProduct.class);
        Mockito.when(fullProduct.getDetails()).thenReturn(new ProductDetails());
        Mockito.when(fullProduct.getAccessories()).thenReturn(new ArrayList());
        Mockito.when(fullProduct.getSimilarItems()).thenReturn(new ArrayList());
        // Act
        fullProduct.addAccessory(miniProduct);
        // Assert
        assertEquals(1, fullProduct.getAccessories().size());
        assertEquals(miniProduct, fullProduct.getAccessories().get(0));
        assertEquals(fullProduct, fullProduct.getSimilarItems().get(0));
    }
}
