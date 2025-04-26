package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class FullProduct_addAccessory_6_2_Test {

    @Test
    public void testAddAccessory() {
        // Create an instance of FullProduct
        FullProduct fullProduct = new FullProduct();
        // Create a MiniProduct object
        MiniProduct miniProduct = new MiniProduct();
        // Call the addAccessory method with the MiniProduct object
        fullProduct.addAccessory(miniProduct);
        // Verify that the accessories list contains the MiniProduct object
        assertEquals(fullProduct.getAccessories().contains(miniProduct), true);
    }
}
