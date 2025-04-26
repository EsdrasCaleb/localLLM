package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
class FullProduct_addAccessory_6_0_Test {

    @Test
    void addAccessory() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = new MiniProduct();
        // Act
        fullProduct.addAccessory(miniProduct);
        // Assert
        List<MiniProduct> actualAccessories = fullProduct.getAccessories();
        assertNotNull(actualAccessories);
        assertTrue(actualAccessories.contains(miniProduct));
    }

    @Test
    void addAccessory_NullInput() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = null;
        // Act
        fullProduct.addAccessory(miniProduct);
        List<MiniProduct> actualAccessories = fullProduct.getAccessories();
        // Assert
        assertNotNull(actualAccessories);
        assertEquals(0, actualAccessories.size());
    }

    @Test
    void addAccessory_EmptyList() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = new MiniProduct();
        // Act
        fullProduct.addAccessory(miniProduct);
        // Assert
        List<MiniProduct> actualAccessories = fullProduct.getAccessories();
        assertNotNull(actualAccessories);
        assertEquals(1, actualAccessories.size());
        assertTrue(actualAccessories.contains(miniProduct));
    }

    // Dummy class for MiniProduct
    static class MiniProduct {

        // Dummy constructor
        public MiniProduct() {
        }
    }

    // Dummy class for FullProduct
    static class FullProduct {

        private List<MiniProduct> accessories = new ArrayList<>();

        public void addAccessory(MiniProduct miniProduct) {
            if (miniProduct != null) {
                accessories.add(miniProduct);
            }
        }

        public List<MiniProduct> getAccessories() {
            return accessories;
        }
    }
}
