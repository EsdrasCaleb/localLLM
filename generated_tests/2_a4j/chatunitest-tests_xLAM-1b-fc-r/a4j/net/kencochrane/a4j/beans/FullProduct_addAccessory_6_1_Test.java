package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FullProduct_addAccessory_6_1_Test {

    @Test
    void addAccessory() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct mockProduct = Mockito.mock(MiniProduct.class);
        // Act
        fullProduct.addAccessory(mockProduct);
        // Assert
        assertTrue(fullProduct.getAccessories().contains(mockProduct));
    }
}
