package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

// Create a new class for the test
public class FullProduct_addAccessory_6_2_Test {

    // Create an instance of the focal class
    private FullProduct fullProduct = new FullProduct();

    @Test
    public void testAddAccessory() {
        // Create a new MiniProduct object
        MiniProduct accessory = new MiniProduct();
        // Call the addAccessory method and pass the MiniProduct object
        fullProduct.addAccessory(accessory);
        // Assert that the accessories ArrayList has the new accessory
        assertTrue(fullProduct.getAccessories().contains(accessory));
    }
}
