package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FullProduct_addAccessory_6_0_Test {

    private FullProduct fullProduct;

    @BeforeEach
    void setUp() {
        fullProduct = new FullProduct();
    }

    @Test
    void testAddAccessory() {
        // Assuming MiniProduct has a default constructor
        MiniProduct miniProduct = new MiniProduct();
        // Verify that the accessories list is initially empty
        assertTrue(fullProduct.getAccessories().isEmpty(), "Accessories list should be empty before adding.");
        // Add the MiniProduct as an accessory
        fullProduct.addAccessory(miniProduct);
        // Verify that the accessories list contains the added MiniProduct
        assertFalse(fullProduct.getAccessories().isEmpty(), "Accessories list should not be empty after adding.");
        assertEquals(1, fullProduct.getAccessories().size(), "Accessories list should contain one item after adding.");
        assertSame(miniProduct, fullProduct.getAccessories().get(0), "The added accessory should be the same MiniProduct instance.");
    }
}
